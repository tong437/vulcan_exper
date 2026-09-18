# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strong JSON-schema and semantic validation at persisted artifact boundaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator


SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "schemas"


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def _validator(schema_name: str) -> Draft202012Validator:
    path = SCHEMA_ROOT / schema_name
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_records(records: list[dict[str, Any]], schema_name: str) -> None:
    """Validate rows structurally and report the first stable row/property failure."""
    validator = _validator(schema_name)
    for index, record in enumerate(records):
        errors = sorted(validator.iter_errors(_native(record)), key=lambda error: list(error.absolute_path))
        if errors:
            location = ".".join(str(part) for part in errors[0].absolute_path) or "<row>"
            raise ValueError(f"{schema_name} row {index} at {location}: {errors[0].message}")


def validate_frame(frame: pd.DataFrame, schema_name: str) -> None:
    """Validate a complete DataFrame and schema-specific cross-column invariants."""
    records = frame.to_dict(orient="records")
    validate_records(records, schema_name)
    if schema_name == "activation_stats.schema.json":
        for index, row in enumerate(records):
            if bool(row["valid"]) and pd.isna(row["activation_rms"]):
                raise ValueError(f"Valid activation row {index} must contain finite activation_rms.")
            if not bool(row["valid"]) and not pd.isna(row["activation_rms"]):
                raise ValueError(f"Invalid activation row {index} must have null activation_rms.")
    elif schema_name == "neuron_scores.schema.json":
        for index, row in enumerate(records):
            if bool(row["valid"]) and any(pd.isna(row[key]) for key in ("activation_rms", "importance", "q")):
                raise ValueError(f"Valid score row {index} must contain finite activation_rms, importance, and q.")
            if not bool(row["valid"]) and not pd.isna(row["q"]):
                raise ValueError(f"Invalid score row {index} must have null q.")
    elif schema_name == "neuron_classes.schema.json":
        expected = {(False, False): "W", (True, False): "V", (False, True): "T", (True, True): "M"}
        for index, row in enumerate(records):
            label = str(row["class"])
            if bool(row["valid"]):
                if label != expected[(bool(row["visual_strong"]), bool(row["text_strong"]))]:
                    raise ValueError(f"Class row {index} disagrees with its visual/text strong bits.")
            elif label != "INVALID":
                raise ValueError(f"Invalid class row {index} must use class=INVALID.")
    elif schema_name == "pairwise_changes.schema.json":
        for index, row in enumerate(records):
            changed = int(row["changed_neurons"])
            valid = int(row["valid_neurons"])
            if changed > valid:
                raise ValueError(f"Pairwise row {index} has more changed than valid neurons.")
            expected_rate = changed / valid if valid else None
            actual_rate = row["change_rate"]
            if expected_rate is None and not pd.isna(actual_rate):
                raise ValueError(f"Pairwise row {index} must have null rate when valid_neurons=0.")
            if expected_rate is not None and not math.isclose(float(actual_rate), expected_rate, abs_tol=1e-12):
                raise ValueError(f"Pairwise row {index} change_rate disagrees with its counts.")
    elif schema_name == "neuron_profiles.schema.json":
        for index, row in enumerate(records):
            counts = [int(row[f"count_{label}"]) for label in ("W", "V", "T", "M")]
            valid_samples = int(row["valid_samples"])
            if sum(counts) != valid_samples:
                raise ValueError(f"Profile row {index} class counts do not sum to valid_samples.")
            proportions = [row[f"proportion_{label}"] for label in ("W", "V", "T", "M")]
            if sum(counts) and not math.isclose(sum(float(value) for value in proportions), 1.0, abs_tol=1e-12):
                raise ValueError(f"Profile row {index} proportions do not sum to one.")
            changed = valid_samples**2 - sum(count**2 for count in counts)
            including_denominator = valid_samples**2
            distinct_denominator = valid_samples * (valid_samples - 1)
            if int(row["changed_ordered_pairs"]) != changed:
                raise ValueError(f"Profile row {index} changed count disagrees with its class counts.")
            if int(row["valid_ordered_pairs_including_self"]) != including_denominator:
                raise ValueError(f"Profile row {index} including-self denominator disagrees with valid_samples.")
            if int(row["valid_distinct_ordered_pairs"]) != distinct_denominator:
                raise ValueError(f"Profile row {index} distinct-pair denominator disagrees with valid_samples.")
            for field, denominator in (
                ("change_rate_including_self", including_denominator),
                ("change_rate_distinct_pairs", distinct_denominator),
            ):
                rate = row[field]
                if denominator and (
                    pd.isna(rate) or not math.isclose(float(rate), changed / denominator, abs_tol=1e-12)
                ):
                    raise ValueError(f"Profile row {index} {field} disagrees with its counts.")
                if not denominator and not pd.isna(rate):
                    raise ValueError(f"Profile row {index} {field} must be null for a zero denominator.")
            dominant = max(proportions)
            if valid_samples and not math.isclose(float(row["dominant_proportion"]), float(dominant), abs_tol=1e-12):
                raise ValueError(f"Profile row {index} dominant_proportion disagrees with class proportions.")
            if not valid_samples and not pd.isna(row["dominant_proportion"]):
                raise ValueError(f"Profile row {index} dominant_proportion must be null without valid samples.")
            total_samples = valid_samples + int(row["invalid_samples"])
            for modality in ("visual", "text"):
                q_count = int(row[f"q_{modality}_valid_samples"])
                if q_count > total_samples:
                    raise ValueError(f"Profile row {index} has more valid q values than total samples.")


def validate_document(document: dict[str, Any], schema_name: str) -> None:
    """Validate one JSON document and summary-level shape invariants."""
    validate_records([document], schema_name)
    if schema_name == "summary_metrics.schema.json":
        including = document["per_layer_change_rate_including_self"]
        distinct = document["per_layer_change_rate_distinct_pairs"]
        if len(including) != len(distinct) or not including:
            raise ValueError("Per-layer summary arrays must be nonempty and have identical lengths.")
        count_arrays = [
            document["per_layer_changed_ordered_comparisons"],
            document["per_layer_valid_ordered_comparisons_including_self"],
            document["per_layer_changed_distinct_ordered_comparisons"],
            document["per_layer_valid_distinct_ordered_comparisons"],
        ]
        if any(len(values) != len(including) for values in count_arrays):
            raise ValueError("Every per-layer rate and count array must have the same length.")
        changed = int(document["overall_changed_ordered_comparisons"])
        changed_distinct = int(document["overall_changed_distinct_ordered_comparisons"])
        if changed != changed_distinct:
            raise ValueError(
                "Including-self and distinct-pair changed counts must match because self-pairs never change."
            )
        including_denominator = int(document["overall_valid_ordered_comparisons_including_self"])
        distinct_denominator = int(document["overall_valid_distinct_ordered_comparisons"])
        including_rate = document["overall_change_rate_including_self"]
        if including_denominator and (
            including_rate is None or not math.isclose(including_rate, changed / including_denominator, abs_tol=1e-12)
        ):
            raise ValueError("Overall including-self rate disagrees with its count denominator.")
        if not including_denominator and including_rate is not None:
            raise ValueError("Overall including-self rate must be null when its denominator is zero.")
        distinct_rate = document["overall_change_rate_distinct_pairs"]
        if distinct_denominator and (
            distinct_rate is None or not math.isclose(distinct_rate, changed / distinct_denominator, abs_tol=1e-12)
        ):
            raise ValueError("Overall distinct-pair rate disagrees with its count denominator.")
        if not distinct_denominator and distinct_rate is not None:
            raise ValueError("Overall distinct-pair rate must be null when its denominator is zero.")
        if sum(count_arrays[0]) != changed or sum(count_arrays[1]) != including_denominator:
            raise ValueError("Overall including-self counts must equal the sum of per-layer counts.")
        if sum(count_arrays[2]) != changed_distinct or sum(count_arrays[3]) != distinct_denominator:
            raise ValueError("Overall distinct-pair counts must equal the sum of per-layer counts.")
        for index, (rate, numerator, denominator) in enumerate(zip(including, count_arrays[0], count_arrays[1])):
            if denominator and (rate is None or not math.isclose(rate, numerator / denominator, abs_tol=1e-12)):
                raise ValueError(f"Layer {index} including-self rate disagrees with its counts.")
            if not denominator and rate is not None:
                raise ValueError(f"Layer {index} including-self rate must be null for a zero denominator.")
        for index, (rate, numerator, denominator) in enumerate(zip(distinct, count_arrays[2], count_arrays[3])):
            if denominator and (rate is None or not math.isclose(rate, numerator / denominator, abs_tol=1e-12)):
                raise ValueError(f"Layer {index} distinct-pair rate disagrees with its counts.")
            if not denominator and rate is not None:
                raise ValueError(f"Layer {index} distinct-pair rate must be null for a zero denominator.")

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
    if schema_name == "neuron_scores.schema.json":
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
            if sum(counts) != int(row["valid_samples"]):
                raise ValueError(f"Profile row {index} class counts do not sum to valid_samples.")
            proportions = [row[f"proportion_{label}"] for label in ("W", "V", "T", "M")]
            if sum(counts) and not math.isclose(sum(float(value) for value in proportions), 1.0, abs_tol=1e-12):
                raise ValueError(f"Profile row {index} proportions do not sum to one.")


def validate_document(document: dict[str, Any], schema_name: str) -> None:
    """Validate one JSON document and summary-level shape invariants."""
    validate_records([document], schema_name)
    if schema_name == "summary_metrics.schema.json":
        including = document["per_layer_change_rate_including_self"]
        distinct = document["per_layer_change_rate_distinct_pairs"]
        if len(including) != len(distinct) or not including:
            raise ValueError("Per-layer summary arrays must be nonempty and have identical lengths.")

# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_structural_pruning import (  # noqa: E402
    bootstrap_latency_comparison,
    validate_generated_token_count,
)
from build_aligned_pruning_artifact import build_aligned_artifacts  # noqa: E402
from build_structural_pruning_artifact import build_artifacts  # noqa: E402
from phase3_structural_utils import (  # noqa: E402
    build_singleton_cluster_idx,
    canonical_json_sha256,
    masks_to_neuron_ids,
    neuron_ids_to_masks,
    theoretical_mlp_parameter_reduction,
    validate_deletion_masks,
    validate_singleton_cluster_idx,
)
from run_phase2_ablation import MLPNeuronAblator  # noqa: E402
from run_phase3_structural import compare_caption_quality, compare_pope_quality  # noqa: E402
from run_phase34_aligned import compare_aligned_hook_caption, compare_aligned_hook_pope  # noqa: E402
from verify_structural_equivalence import (  # noqa: E402
    gate_bf16_hook_comparison,
    gate_exact_reload_comparison,
)

from llamafactory.train.vulcan.pruning import pruning_mlp  # noqa: E402


class TinyMLP(torch.nn.Module):
    def __init__(self, hidden_size: int = 3, intermediate_size: int = 8):
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()

    def forward(self, x):
        return x + self.mlp(x)


class TinyModel(torch.nn.Module):
    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyLayer() for _ in range(num_layers)])
        self.config = type("Config", (), {"intermediate_size": 8})()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _score_table(num_layers: int = 2, width: int = 20) -> pd.DataFrame:
    rows = []
    for layer in range(num_layers):
        for neuron in range(width):
            rows.append(
                {
                    "layer": layer,
                    "neuron_idx": neuron,
                    "q_visual": 0.0,
                    "q_text": 0.0,
                    "q_multimodal": float(width - neuron),
                    "q_unknown": 0.0,
                    "r_visual": 0.0,
                    "r_text": 0.0,
                    "r_multimodal": float(neuron),
                    "r_unknown": 0.0,
                    "is_dead": False,
                }
            )
    return pd.DataFrame(rows)


def test_frozen_artifact_has_exact_band_counts_and_singleton_coverage():
    masks, summary, cluster_idx, details = build_artifacts(
        _score_table(),
        band_start=0.05,
        band_end=0.20,
        expected_num_layers=2,
        expected_layer_width=20,
        hidden_size=3,
    )

    assert summary["selected_neurons"] == 6
    assert summary["per_layer_selected"] == {"0": 3, "1": 3}
    assert masks_to_neuron_ids(masks) == {"0": [1, 2, 3], "1": [1, 2, 3]}
    assert len(cluster_idx) == 2
    assert len(cluster_idx[0]) == 17
    assert cluster_idx[0][0] == {"anchor": 0, "neuron": [0]}
    assert cluster_idx[0][1] == {"anchor": 4, "neuron": [4]}
    assert details["selected_dead_neurons"] == []


def test_aligned_artifact_uses_exact_rank_window_and_target_width():
    masks, summary, cluster_idx, details = build_aligned_artifacts(
        _score_table(width=20),
        rank_start=1,
        rank_count=4,
        expected_num_layers=2,
        expected_layer_width=20,
        hidden_size=3,
    )

    assert masks_to_neuron_ids(masks) == {"0": [1, 2, 3, 4], "1": [1, 2, 3, 4]}
    assert summary["per_layer_selected"] == {"0": 4, "1": 4}
    assert len(cluster_idx[0]) == 16
    assert details["target_intermediate_size"] == 16
    assert details["rank_end"] == 5


def test_mask_roundtrip_validation_and_hash_are_deterministic():
    ids = {"0": [1, 3], "1": [0, 7]}
    masks = neuron_ids_to_masks(ids, {0: 8, 1: 8})
    summary = validate_deletion_masks(
        masks, expected_num_layers=2, expected_layer_width=8, expected_pruned_per_layer=2
    )
    clusters = build_singleton_cluster_idx(masks)

    assert summary["total_pruned"] == 4
    assert masks_to_neuron_ids(masks) == ids
    assert validate_singleton_cluster_idx(clusters, masks)["validated"]
    assert canonical_json_sha256(ids) == canonical_json_sha256({"1": [0, 7], "0": [1, 3]})
    assert theoretical_mlp_parameter_reduction(masks, hidden_size=3)["removed_parameters"] == 36


def test_mask_validation_rejects_duplicate_and_out_of_range_ids():
    with pytest.raises(ValueError, match="duplicate"):
        neuron_ids_to_masks({"0": [1, 1]}, {0: 4})
    with pytest.raises(ValueError, match="out-of-range"):
        neuron_ids_to_masks({"0": [4]}, {0: 4})


def test_singleton_structural_pruning_matches_hook_ablation():
    torch.manual_seed(7)
    original = TinyModel(num_layers=2).eval()
    original.requires_grad_(False)
    structural = copy.deepcopy(original)
    masks = neuron_ids_to_masks({"0": [1, 3], "1": [0, 7]}, {0: 8, 1: 8})
    inputs = torch.randn(2, 5, 3)

    with MLPNeuronAblator(original, masks):
        hook_output = original(inputs)
    summary = pruning_mlp(structural, build_singleton_cluster_idx(masks))
    structural_output = structural(inputs)

    assert summary.original_intermediate_size == 8
    assert summary.pruned_intermediate_size == 6
    assert not any(parameter.requires_grad for parameter in structural.parameters())
    assert torch.allclose(hook_output, structural_output, atol=1e-6, rtol=1e-6)


def test_phase3_quality_gates_compare_structural_with_hook_and_original():
    caption_reference = {
        "metrics": {
            "none": {"nll": 4.1},
            "rank_band:multimodal:0.05:0.2": {"nll": 3.9},
        }
    }
    caption_structural = {"metrics": {"none": {"nll": 3.9005}}}
    assert compare_caption_quality(caption_reference, caption_structural)["passed"]

    predictions = [
        {"source_index": 0, "prediction": "yes", "margin": 1.0},
        {"source_index": 1, "prediction": "no", "margin": -1.0},
    ]
    pope_reference = {
        "metrics": {
            "none": {"accuracy": 0.90, "f1": 0.89, "yes_ratio": 0.50},
            "rank_band:multimodal:0.05:0.2": {
                "accuracy": 0.895,
                "f1": 0.885,
                "yes_ratio": 0.49,
                "predictions": predictions,
            },
        }
    }
    pope_structural = {
        "metrics": {
            "none": {
                "accuracy": 0.895,
                "f1": 0.885,
                "yes_ratio": 0.49,
                "predictions": predictions,
            }
        }
    }
    result = compare_pope_quality(pope_reference, pope_structural)
    assert result["passed"]
    assert result["prediction_match_ratio"] == 1.0


def test_aligned_hook_quality_gate_requires_task_safety_and_baseline_reproduction():
    caption_reference = {
        "metrics": {
            "none": {"nll": 4.1},
            "rank_band:multimodal:0.05:0.2": {"nll": 3.9},
        }
    }
    caption_aligned = {
        "metrics": {
            "none": {"nll": 4.1015},
            "rank_window:multimodal:180:512": {"nll": 3.76},
        }
    }
    caption_result = compare_aligned_hook_caption(caption_reference, caption_aligned)
    assert caption_result["passed"]
    assert caption_result["aligned_minus_frozen_qband_nll"] < -0.05

    baseline_predictions = [
        {"source_index": 0, "prediction": "yes"},
        {"source_index": 1, "prediction": "no"},
    ]
    pope_reference = {
        "metrics": {
            "none": {
                "accuracy": 0.90,
                "f1": 0.89,
                "yes_ratio": 0.50,
                "predictions": baseline_predictions,
            }
        }
    }
    pope_aligned = {
        "metrics": {
            "none": {
                "accuracy": 0.90,
                "f1": 0.89,
                "yes_ratio": 0.50,
                "predictions": baseline_predictions,
            },
            "rank_window:multimodal:180:512": {
                "accuracy": 0.895,
                "f1": 0.885,
                "yes_ratio": 0.49,
                "predictions": baseline_predictions,
            },
        }
    }
    assert compare_aligned_hook_pope(pope_reference, pope_aligned)["passed"]


def test_phase3_pope_gate_accepts_rare_bf16_boundary_flip_but_reports_it():
    hook_predictions = [{"source_index": row, "prediction": "yes", "margin": 1.0} for row in range(1000)]
    structural_predictions = copy.deepcopy(hook_predictions)
    hook_predictions[17].update({"prediction": "no", "margin": 0.0})
    structural_predictions[17].update({"prediction": "yes", "margin": 0.125})
    reference = {
        "metrics": {
            "none": {"accuracy": 0.90, "f1": 0.89, "yes_ratio": 0.50},
            "rank_band:multimodal:0.05:0.2": {
                "accuracy": 0.895,
                "f1": 0.885,
                "yes_ratio": 0.49,
                "predictions": hook_predictions,
            },
        }
    }
    structural = {
        "metrics": {
            "none": {
                "accuracy": 0.896,
                "f1": 0.886,
                "yes_ratio": 0.491,
                "predictions": structural_predictions,
            }
        }
    }

    result = compare_pope_quality(reference, structural)

    assert result["passed"]
    assert not result["exact_prediction_match"]
    assert result["prediction_mismatch_count"] == 1
    assert result["prediction_mismatches"][0]["near_decision_boundary"]
    assert result["boundary_diagnostic"]["all_mismatches_within_reference_margin"]


def test_phase3_pope_gate_rejects_high_margin_prediction_flip():
    hook_predictions = [{"source_index": 0, "prediction": "yes", "margin": 1.0}]
    structural_predictions = [{"source_index": 0, "prediction": "no", "margin": -1.0}]
    reference = {
        "metrics": {
            "none": {"accuracy": 0.90, "f1": 0.89, "yes_ratio": 0.50},
            "rank_band:multimodal:0.05:0.2": {
                "accuracy": 0.90,
                "f1": 0.89,
                "yes_ratio": 0.50,
                "predictions": hook_predictions,
            },
        }
    }
    structural = {
        "metrics": {
            "none": {
                "accuracy": 0.90,
                "f1": 0.89,
                "yes_ratio": 0.50,
                "predictions": structural_predictions,
            }
        }
    }

    result = compare_pope_quality(reference, structural)

    assert not result["passed"]
    assert not result["checks"]["prediction_agreement"]
    assert not result["boundary_diagnostic"]["all_mismatches_within_reference_margin"]
    assert not result["boundary_diagnostic"]["is_hard_gate"]


def test_bf16_equivalence_gate_treats_pointwise_gemm_drift_as_diagnostic():
    metrics = {
        "mean_label_nll_delta": 0.005,
        "prediction_match_ratio": 1.0,
        "logit_max_abs": 0.75,
        "candidate_logprob_max_abs": 0.533,
    }
    args = SimpleNamespace(
        nll_tolerance=0.05,
        logit_max_abs_tolerance=1.0,
        candidate_logprob_tolerance=0.5,
    )

    result = gate_bf16_hook_comparison(metrics, args)

    assert result["passed"]
    assert not result["diagnostic_bounds"]["candidate_logprob_max_abs"]["within_reference_bound"]


def test_reload_equivalence_gate_requires_exact_outputs():
    exact_metrics = {
        "logit_max_abs": 0.0,
        "candidate_logprob_max_abs": 0.0,
        "max_per_sample_label_nll_delta": 0.0,
        "prediction_match_ratio": 1.0,
    }
    assert gate_exact_reload_comparison(exact_metrics)["passed"]

    drifted_metrics = {**exact_metrics, "logit_max_abs": 1e-7}
    assert not gate_exact_reload_comparison(drifted_metrics)["passed"]


def test_benchmark_latency_comparison_reports_direct_speedup_ci():
    result = bootstrap_latency_comparison(
        [2.0, 2.1, 1.9, 2.0],
        [1.0, 1.1, 0.9, 1.0],
        bootstrap_samples=500,
        seed=7,
    )

    assert result["latency_speedup"] == pytest.approx(2.0)
    assert result["latency_speedup_ci95"][0] > 1.0
    assert result["latency_delta_seconds_ci95"][1] < 0.0


def test_benchmark_validates_actual_fixed_generation_length():
    class FixedLengthModel:
        def generate(self, input_ids, max_new_tokens, **kwargs):
            del kwargs
            suffix = torch.zeros((input_ids.shape[0], max_new_tokens), dtype=input_ids.dtype)
            return torch.cat([input_ids, suffix], dim=1)

    kwargs = {"input_ids": torch.ones((1, 3), dtype=torch.long), "max_new_tokens": 8}
    assert validate_generated_token_count(FixedLengthModel(), kwargs, expected=8) == 8

    with pytest.raises(RuntimeError, match="expected 7"):
        validate_generated_token_count(FixedLengthModel(), kwargs, expected=7)

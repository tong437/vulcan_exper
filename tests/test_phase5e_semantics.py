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

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from phase5e_semantics import evaluate_caption_semantics, extract_final_answer  # noqa: E402


def test_extract_final_answer_ignores_closed_thinking():
    result = extract_final_answer("<think>long reasoning</think>\n\nCaption: Several bicycles are inside a train car.")
    assert result["final_answer"] == "Several bicycles are inside a train car."
    assert result["had_thinking"]
    assert not result["unclosed_thinking"]


def test_unclosed_or_thinking_only_output_fails():
    unclosed = evaluate_caption_semantics("<think>Several bikes appear to be in a train")
    thinking_only = evaluate_caption_semantics("<think>reasoning</think>")
    assert unclosed["verdict"] == "fail"
    assert "unclosed_thinking" in unclosed["reasons"]
    assert thinking_only["verdict"] == "fail"
    assert "no_final_answer" in thinking_only["reasons"]


def test_contract_accepts_semantically_equivalent_captions():
    captions = (
        "Several bicycles are parked inside a train car.",
        "A group of bikes can be seen on a subway.",
        "Multiple bikes are lined up in a railway carriage.",
        "A subway train with several bicycles stored inside.",
    )
    for caption in captions:
        result = evaluate_caption_semantics(caption)
        assert result["verdict"] == "pass", (caption, result)


def test_contract_rejects_explicit_failures():
    captions = (
        "There are no bicycles inside the train.",
        "Only one bicycle is parked on a subway train.",
        "Several bicycles are parked outdoors on the street.",
        "An interior view of a train or bus with two bicycles.",
        "Bicycle train",
    )
    for caption in captions:
        assert evaluate_caption_semantics(caption)["verdict"] == "fail"


def test_contract_routes_incomplete_semantics_to_review():
    result = evaluate_caption_semantics("Several bicycles are parked beside the windows.")
    assert result["verdict"] == "review"
    assert not result["automatic_pass"]
    assert result["requires_human_review"]


def test_token_limit_is_a_hard_failure_even_when_semantics_are_present():
    result = evaluate_caption_semantics(
        "Several bicycles are parked inside a train car.",
        terminated_normally=False,
        hit_token_limit=True,
    )
    assert result["verdict"] == "fail"
    assert "generation_not_normally_terminated" in result["reasons"]

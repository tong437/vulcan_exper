# Copyright 2025 the LlamaFactory team.
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

"""Frozen prompt construction for the advisor experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def mmbench_selection_prompt(sample: Mapping[str, Any], replay_row: Mapping[str, Any] | None) -> str:
    """Return exactly the MMBench question and valid choices, excluding hint and generation instructions."""
    if replay_row is None:
        raise ValueError("MMBench requires the frozen replay row with choices.")
    question = str(sample["question"]).strip()
    choices = replay_row.get("choices", {})
    if not isinstance(choices, Mapping):
        raise ValueError(f"MMBench sample {sample['sample_id']} has malformed choices.")
    valid_choices = [
        f"{label}. {str(text).strip()}"
        for label, text in choices.items()
        if text is not None and str(text).strip() and str(text).strip().lower() != "nan"
    ]
    if not valid_choices:
        raise ValueError(f"MMBench sample {sample['sample_id']} has no usable choices.")
    return "\n\n".join([question, *valid_choices])

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

from neuron_typing_generalization.clevr_audit import program_edit_distance


def _node(function: str, value: str | None = None) -> dict:
    return {"function": function, "inputs": [], "value_inputs": [] if value is None else [value]}


def test_program_edit_distance_covers_literal_substitution_and_operation_insertion():
    base = [_node("scene"), _node("filter_color", "red"), _node("count")]
    assert program_edit_distance(base, base) == 0
    assert program_edit_distance(base, [_node("scene"), _node("filter_color", "blue"), _node("count")]) == 1
    assert program_edit_distance(base, [*base, _node("exist")]) == 1

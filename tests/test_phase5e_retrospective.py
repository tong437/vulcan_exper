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

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from run_phase5e_retrospective import parse_checkpoint_spec  # noqa: E402


def test_parse_checkpoint_spec_preserves_paths_with_equals():
    name, path = parse_checkpoint_spec("core13=/tmp/model=copy")
    assert name == "core13"
    assert path == Path("/tmp/model=copy")


@pytest.mark.parametrize("value", ["missing_separator", "=path", "name="])
def test_parse_checkpoint_spec_rejects_malformed_values(value):
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_checkpoint_spec(value)

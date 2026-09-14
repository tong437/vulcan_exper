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

from neuron_typing_generalization.benchmark import run_scale_benchmark


def test_scale_benchmark_is_chunk_invariant_at_fixture_dimensions():
    report = run_scale_benchmark(sample_count=8, layer_count=2, d_ff=16, chunk_size=3)
    assert report["status"] == "passed"
    assert report["shape"] == {"samples": 8, "layers": 2, "d_ff": 16}
    assert report["stability_shape"] == [2, 16, 4]

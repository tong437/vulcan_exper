#!/usr/bin/env bash
# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec /root/miniconda3/bin/python -u research/neuron_typing_generalization/pipelines/orchestrate_throughput_256.py --gpu "${1:-0}" --model gemma_4_e2b_it

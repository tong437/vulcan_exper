# Layerwise V/T/M/C/W response-state distributions

All 16 runs use the exact answer-generation prompts from the preceding ablation experiment and
`add_generation_prompt=True`. `C` contains non-visual, non-question-text, non-padding decoder positions. Every layer
is averaged over 1,000 frozen inputs and its real FFN neurons; no layers or neurons are padded.

| Model | Dataset | V | T | M | C | W |
|---|---|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | MME | 18.84% | 18.87% | 14.86% | 10.56% | 36.87% |
| LLaVA-1.5-7B | MMBench | 18.54% | 17.26% | 15.10% | 10.98% | 38.12% |
| LLaVA-1.5-7B | TextVQA | 18.59% | 19.17% | 15.02% | 10.71% | 36.52% |
| LLaVA-1.5-7B | ChartQA | 18.04% | 18.55% | 15.30% | 11.03% | 37.07% |
| Qwen3-VL-2B | MME | 17.48% | 17.07% | 13.97% | 9.24% | 42.22% |
| Qwen3-VL-2B | MMBench | 16.43% | 16.11% | 14.81% | 10.04% | 42.62% |
| Qwen3-VL-2B | TextVQA | 17.51% | 17.14% | 13.91% | 9.08% | 42.36% |
| Qwen3-VL-2B | ChartQA | 16.03% | 16.01% | 15.34% | 9.66% | 42.97% |
| Qwen3.5-2B | MME | 17.23% | 18.88% | 16.28% | 12.31% | 35.30% |
| Qwen3.5-2B | MMBench | 16.88% | 16.98% | 17.22% | 13.21% | 35.72% |
| Qwen3.5-2B | TextVQA | 16.32% | 19.40% | 16.37% | 11.86% | 36.06% |
| Qwen3.5-2B | ChartQA | 15.40% | 17.56% | 17.74% | 12.89% | 36.41% |
| Gemma-4-E2B-it | MME | 14.85% | 16.32% | 14.25% | 8.07% | 46.51% |
| Gemma-4-E2B-it | MMBench | 14.54% | 16.32% | 14.34% | 8.51% | 46.29% |
| Gemma-4-E2B-it | TextVQA | 13.80% | 16.06% | 14.66% | 8.23% | 47.25% |
| Gemma-4-E2B-it | ChartQA | 13.39% | 15.69% | 15.07% | 8.61% | 47.24% |

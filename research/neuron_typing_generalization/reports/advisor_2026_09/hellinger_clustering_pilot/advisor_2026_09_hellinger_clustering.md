# Hellinger average-linkage neuron clustering pilot

This is an explicitly sampled pilot, not a full-neuron clustering result. Within each model × dataset run, at most
4,096 real neurons are selected without replacement using deterministic proportional
layer-stratified sampling. Their exact all-pairs Hellinger distances are clustered by standard average linkage and cut
at `d_H = 0.20`. Cluster IDs are canonicalized by decreasing cluster size.

| Model | Dataset | Sampled neurons | Clusters | Largest cluster | Full distance storage |
|---|---|---:|---:|---:|---:|
| LLaVA-1.5-7B | MME | 4,096 | 22 | 22.1% | 462.2 GiB |
| LLaVA-1.5-7B | MMBench | 4,096 | 23 | 34.4% | 462.2 GiB |
| LLaVA-1.5-7B | TextVQA | 4,096 | 22 | 26.8% | 462.2 GiB |
| LLaVA-1.5-7B | ChartQA | 4,096 | 23 | 29.5% | 462.2 GiB |
| Qwen3-VL-2B | MME | 4,096 | 28 | 18.7% | 110.2 GiB |
| Qwen3-VL-2B | MMBench | 4,096 | 25 | 20.1% | 110.2 GiB |
| Qwen3-VL-2B | TextVQA | 4,096 | 23 | 22.8% | 110.2 GiB |
| Qwen3-VL-2B | ChartQA | 4,096 | 28 | 22.8% | 110.2 GiB |
| Qwen3.5-2B | MME | 4,096 | 26 | 18.1% | 81.0 GiB |
| Qwen3.5-2B | MMBench | 4,096 | 24 | 24.6% | 81.0 GiB |
| Qwen3.5-2B | TextVQA | 4,096 | 24 | 15.9% | 81.0 GiB |
| Qwen3.5-2B | ChartQA | 4,096 | 22 | 21.2% | 81.0 GiB |
| Gemma-4-E2B-it | MME | 4,096 | 27 | 32.2% | 425.4 GiB |
| Gemma-4-E2B-it | MMBench | 4,096 | 22 | 24.9% | 425.4 GiB |
| Gemma-4-E2B-it | TextVQA | 4,096 | 25 | 33.2% | 425.4 GiB |
| Gemma-4-E2B-it | ChartQA | 4,096 | 24 | 33.4% | 425.4 GiB |

The last column is the float64 condensed distance vector alone and excludes linkage working memory. Per-run sampled
assignments, linkage matrices, cluster centroids and integrity hashes are stored in the sibling run directories.
Presentation and any full-scale approximation method remain deliberately unspecified.

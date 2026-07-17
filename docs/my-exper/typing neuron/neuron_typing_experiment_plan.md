# Neuron Typing & Type-Aware Pruning for Qwen3.5-VL-0.8B

## Research Question

> In a hybrid linear/full-attention VLM decoder, are multimodal FFN neurons preferentially formed around full-attention layers, and can this structure guide safer pruning?

### Hypotheses

- **H1**: Qwen3.5-VL-0.8B's language decoder FFN neurons exhibit modality functional specialization (visual / text / multimodal / unknown), with a layer-level structure.
- **H2**: These functional types have causal significance — ablating visual neurons hurts visual perception, ablating text neurons hurts text ability, ablating multimodal neurons hurts cross-modal reasoning, and ablating unknown neurons is relatively safe.
- **H3**: Protecting multimodal neurons and preferentially pruning unknown neurons yields better VLM pruning outcomes than random / magnitude / activation-based pruning at the same ratio.

---

## Model Architecture Summary

| Component | Detail |
|-----------|--------|
| Model | `Qwen/Qwen3.5-0.8B` (`Qwen3_5ForConditionalGeneration`) |
| Local path | `/root/autodl-pub-RTX4090-hdd-1/models/qwen3.5-0.8b` |
| Text backbone | 24 layers, hidden_size=1024, intermediate_size=3584 |
| MLP type | SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))` |
| Total FFN neurons | 24 × 3584 = **86,016** |
| Attention types | 6 Full Attention layers (L3, L7, L11, L15, L19, L23) + 18 GatedDeltaNet linear-attention layers |
| Vision encoder | 12-layer ViT, hidden_size=768, patch_size=16, spatial_merge_size=2, projects to 1024-dim |
| Special tokens | `image_token_id=248056`, `vision_start=248053`, `vision_end=248054` (read from config at runtime) |
| Template | `qwen3_5_nothink` |

### Attention Block Structure (for blocked permutation)

```
Block 1: L0(GDN), L1(GDN), L2(GDN), L3(FA)    → depth 0-3
Block 2: L4(GDN), L5(GDN), L6(GDN), L7(FA)    → depth 4-7
Block 3: L8(GDN), L9(GDN), L10(GDN), L11(FA)   → depth 8-11
Block 4: L12(GDN), L13(GDN), L14(GDN), L15(FA)  → depth 12-15
Block 5: L16(GDN), L17(GDN), L18(GDN), L19(FA)  → depth 16-19
Block 6: L20(GDN), L21(GDN), L22(GDN), L23(FA)  → depth 20-23
```

Each block has 1 FA layer + 3 GDN layers. Blocked permutation swaps FA/GDN label within each block.

---

## Phase 1: Neuron Typing

### 1.1 Neuron Definition

A neuron is defined as the j-th dimension of the intermediate activation in a SwiGLU MLP layer:

```
gate = gate_proj(hidden_states)
up   = up_proj(hidden_states)
intermediate = silu(gate) * up    ← neuron j = intermediate[..., j]
output = down_proj(intermediate)
```

Hook target: **`down_proj` forward-pre-hook**, capturing `inputs[0]` which is exactly `silu(gate) * up`.

Path: `model.language_model.layers[l].mlp.down_proj`

### 1.2 Data

| Phase | Dataset | Size | Format |
|-------|---------|------|--------|
| Pilot | COCO Captions val2017 subset | 500-1,000 | image + caption |
| Formal | COCO Captions train2017 + val2017 | 5,000 | image + caption |

#### Input Construction

```
User: <image>
Assistant: {caption}
```

No fixed template text beyond the minimal `User:` / `Assistant:` markers. The caption text varies per sample.

#### Token Masks

Three masks per sample, recorded at construction time (not post-hoc string matching):

| Mask | Definition |
|------|-----------|
| `visual_mask` | `input_ids == image_token_id` |
| `caption_mask` | tokens in the caption span (recorded start/end position at construction) |
| `ignore_mask` | everything else (system/user prompt, special tokens, `<|im_start|>`, etc.) |

Special token IDs are read from config at runtime:
```python
image_token_id = getattr(model.config, "image_token_id", None)
# fallback to tokenizer
vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
vision_end   = tokenizer.convert_tokens_to_ids("<|vision_end|>")
```

For future VQA extension, `caption_mask` can be split into `question_mask` + `answer_mask`, but Phase 1 uses image-caption pairs only.

### 1.3 Activation Collection: Two-Pass Protocol

#### Pass 1: Global Max per Neuron

For each sample, each layer, each neuron, record the maximum positive activation:

```
a = intermediate_activation[sample, layer]    # [seq_len, 3584]
a_pos = max(a, 0)
global_max[layer] = max(global_max[layer], a_pos.max(dim=seq_len))
```

`global_max[layer]` shape: `[3584]`. Accumulated across all samples.

#### Pass 2: Per-Sample Visual/Text Activation Count

For each sample, each layer:

```
a_norm = clamp(a, min=0) / global_max[layer] * 10

visual_count[j] = number of visual tokens where a_norm[token, j] > T_visual
text_count[j]   = number of caption tokens where a_norm[token, j] > T_text
```

A neuron is classified per-sample:

| visual_count > n_visual | text_count > n_text | Sample type |
|:---:|:---:|---|
| Yes | No | **visual** |
| No | Yes | **text** |
| Yes | Yes | **multimodal** |
| No | No | **unknown** |

### 1.4 Threshold Calibration

**Do NOT directly use LLaVA paper thresholds.** Run a pilot first.

#### Pilot Protocol (500-1,000 samples)

1. Collect Pass 1 global max
2. For each layer, plot activation distribution histograms (visual tokens vs text tokens)
3. Determine T_visual, T_text, n_visual, n_text from actual distribution

#### Threshold Approaches (compare both)

| Approach | Definition |
|----------|-----------|
| Absolute | `a_norm > T` (fixed T across all layers/neurons) |
| Quantile | `a_norm > per-neuron q95` or `q97` |

If both approaches yield consistent layer-level trends, the typing is robust.

#### Threshold Sweep (formal experiment)

| Parameter | Values |
|-----------|--------|
| T_visual | {1.5, 2.0, 2.5} |
| n_visual | {2, 4, 8} |
| T_text | {2.0, 3.0, 4.0} |
| n_text | {1, 2, 4} |

Report: (1) 4-class neuron distribution stability across thresholds, (2) high-confidence neuron count stability, (3) downstream ablation ranking stability.

### 1.5 Top-K Sample Scoring

For each neuron j in each layer, define the per-sample activation score:

```
sample_score[j] = max(a_norm[token, j]) over visual + caption tokens
```

Keep top-K samples per neuron. Test K ∈ {20, 30, 50}.

If Spearman correlation of neuron type rankings across K values is high (ρ > 0.8), the typing is K-stable.

For each neuron, compute:

```
p_visual     = (top-K 里 visual samples) / K
p_text       = (top-K 里 text samples) / K
p_multimodal = (top-K 里 multimodal samples) / K
p_unknown    = (top-K 里 unknown samples) / K
```

### 1.6 Statistical Comparison: FA vs GDN Layers

**Core question**: Do FFN neurons after Full Attention layers have different type distributions than those after GatedDeltaNet layers?

#### Blocked Permutation Test

Each block = one FA layer + its adjacent GDN layers (see attention block structure above). For each neuron type:

```python
# block-level ratios
blocks = [
    (fa_ratio_L3,  mean(gd_ratios_L0_L1_L2)),
    (fa_ratio_L7,  mean(gd_ratios_L4_L5_L6)),
    (fa_ratio_L11, mean(gd_ratios_L8_L9_L10)),
    (fa_ratio_L15, mean(gd_ratios_L12_L13_L14)),
    (fa_ratio_L19, mean(gd_ratios_L16_L17_L18)),
    (fa_ratio_L23, mean(gd_ratios_L20_L21_L22)),
]

fa_obs = [b[0] for b in blocks]
gd_obs = [b[1] for b in blocks]
observed_diff = mean(fa_obs) - mean(gd_obs)

# Blocked permutation: swap FA/GDN label within each block
for _ in range(10000):
    perm_fa, perm_gd = [], []
    for fa_r, gd_r in blocks:
        if random() < 0.5:
            perm_fa.append(fa_r); perm_gd.append(gd_r)
        else:
            perm_fa.append(gd_r); perm_gd.append(fa_r)
    perm_diffs.append(mean(perm_fa) - mean(perm_gd))

p_value = mean(|perm_diffs| >= |observed_diff|)
```

**Note**: n_blocks = 6, so 2^6 = 64 possible permutations. p-value resolution floor = 1/64 ≈ 0.016. Report this limitation.

#### Report Format

For each neuron type (visual, text, multimodal, unknown):

```
FA multimodal ratio:  12.4%
GDN multimodal ratio:  7.1%
Difference:           +5.3 pp
Blocked permutation p = 0.031 (n=6, floor=0.016)
95% bootstrap CI:     [+2.1 pp, +8.4 pp]
```

Report **observed difference**, **permutation p-value**, and **effect size CI** (layer-level bootstrap).

### 1.7 Outputs: Phase 1

| Output | Format | Description |
|--------|--------|-------------|
| `neuron_type_scores.parquet` | DataFrame | Per-neuron: layer, index, p_visual, p_text, p_multimodal, p_unknown, attention_type (FA/GDN) |
| `fig_layer_distribution.png` | Figure | Per-layer count of 4 neuron types (high-confidence threshold: p_type >= 0.7), FA/GDN layers color-coded |
| `fig_layer_ratio.png` | Figure | Per-layer ratio of 4 neuron types |
| `fig_scatter.png` | Figure | Scatter: x=p_visual, y=p_text, color=p_multimodal |
| `fig_fa_vs_gdn.png` | Figure | Bar chart: mean type ratios in FA vs GDN layers with CI error bars |
| `fig_threshold_sensitivity.png` | Figure | How neuron counts change across threshold sweep |
| `fig_top_examples.png` | Figure | Top-activated examples per neuron type (10 neurons each) |
| `perm_test_results.json` | JSON | Per-type: observed_diff, p_value, ci_lo, ci_hi |

---

## Phase 2: Causal Ablation

### 2.1 Ablation Mechanism

Register forward-pre-hook on `down_proj`:

```python
def make_hook(neuron_indices):
    def hook(module, inputs):
        intermediate = inputs[0]           # [batch, seq, 3584]
        intermediate[..., neuron_indices] = 0
        return (intermediate,)             # return modified input
    return hook
```

Do NOT modify weights. Dynamic hook-based ablation is safer and reversible.

### 2.2 Ablation Groups

| Group | Selection | Purpose |
|-------|-----------|---------|
| A. visual neurons | p_visual >= 0.7, top by p_visual | Test visual perception causality |
| B. text neurons | p_text >= 0.7, top by p_text | Test text ability causality |
| C. multimodal neurons | p_multimodal >= 0.7, top by p_multimodal | Test cross-modal reasoning causality |
| D. unknown neurons | p_unknown >= 0.7, top by p_unknown | Test low-risk pruning candidate |
| E. random neurons | Random from all layers | Baseline |
| F. **layer-type-matched random** | Random from same FA/GDN layer distribution | **Critical control** |
| G. layer-matched random | Random from same per-layer counts | Control for depth |
| H. low-magnitude neurons | Lowest mean |activation| | Traditional pruning baseline |

**Group F is mandatory.** If ablating multimodal neurons (which concentrate in FA layers), the random baseline must also sample from the same FA/GDN distribution. Otherwise the difference could be "FA layers are important" not "multimodal neurons are important."

### 2.3 Ablation Ratios

| Ratio | Neurons (of 86,016) | Role |
|-------|---------------------|------|
| 0.1% | ~86 | Minimal perturbation |
| 0.3% | ~258 | |
| 0.5% | ~430 | Main result |
| 1.0% | ~860 | Main result |
| 2.0% | ~1,720 | Main result |
| 3.0% | ~2,580 | Main result |
| 5.0% | ~4,300 | Stress test |

Each ratio: same number of neurons from each type group AND matched random baseline, respecting per-layer proportions.

### 2.4 Evaluation Tasks

#### Visual Perception (expect visual neuron ablation hurts most)

| Task | Metric | Notes |
|------|--------|-------|
| COCO Caption (1k subset) | BLEU-4, CIDEr | |
| POPE (hallucination) | Accuracy, F1 | Object existence判断 |

#### Cross-Modal Reasoning (expect multimodal neuron ablation hurts most)

| Task | Metric | Notes |
|------|--------|-------|
| VQAv2 (2k subset) | VQA Accuracy | |
| GQA (1k subset) | Accuracy | Spatial/relational reasoning |

#### Text-Only (expect text neuron ablation hurts most)

| Task | Metric | Notes |
|------|--------|-------|
| WikiText-103 | Perplexity | |
| HellaSwag (1k subset) | Accuracy | Commonsense |

#### Baseline Sanity Checks

| Check | Purpose |
|-------|---------|
| Image-present vs image-removed on same VQA samples | Verify model actually uses image input |
| Full model (no ablation) on all tasks | Baseline scores |

**Run baselines first.** If any task baseline is near-random (e.g., HellaSwag < 30% for 0.8B), drop that task.

### 2.5 Same-Layer Swap Ablation

For each block (e.g., layers 20-23), compare:

- Ablate 1000 high-confidence multimodal neurons
- Ablate 1000 high-confidence unknown neurons from same layers
- Ablate 1000 random neurons from same layers

This is the cleanest causal test: same layer, same count, different neuron type.

### 2.6 Expected Results Pattern

| Ablation target | Caption/Perception | VQA/Reasoning | Text-only |
|----------------|-------------------|---------------|-----------|
| visual neurons | **Large drop** | Medium drop | Small drop |
| text neurons | Small/medium drop | Medium drop | **Large drop** |
| multimodal neurons | Medium drop | **Large drop** | Small/medium drop |
| unknown neurons | **Minimal drop** | **Minimal drop** | **Minimal drop** |
| random neurons | Medium drop | Medium drop | Medium drop |
| layer-matched random | Medium drop | Medium drop | Medium drop |

If this selective degradation pattern holds, H2 is supported.

### 2.7 Outputs: Phase 2

| Output | Format |
|--------|--------|
| `ablation_results.jsonl` | Per-group, per-ratio, per-task: score, delta from baseline |
| `fig_selective_ablation.png` | Heatmap: ablation group × task, color = delta |
| `fig_ablation_ratio_curve.png` | Line chart: ratio × score for each group |
| `fig_same_layer_swap.png` | Bar chart: same-layer swap results |

---

## Phase 3: Type-Aware Pruning

### 3.1 Pruning Mode

**Phase 3A: Masked pruning (functional pruning)**

```python
intermediate[..., pruned_neurons] = 0
```

Does not reduce parameters but validates pruning selection. This is the primary mode for Phase 3.

**Phase 3B: Structural pruning** (only after Phase 3A results are stable)

Remove rows/columns from gate_proj, up_proj, down_proj. Use existing `VulcanQwen3_5ForConditionalGeneration` for non-uniform checkpoint loading. Deferred to later.

### 3.2 Pruning Score

```python
prune_score = (
    p_unknown
    - 1.0 * p_multimodal      # strongest protection
    - 0.3 * p_visual
    - 0.3 * p_text
    - 0.3 * freq_high          # fraction of tokens where normalized activation > T_freq
    - 0.3 * mean_abs_activation # normalized mean |activation|
)
```

**Simplified first version**: `prune_score = p_unknown - p_multimodal`

Where:
- `freq_high` = fraction of tokens where normalized activation > T_freq (NOT >0, which is too broad for SwiGLU)
- `mean_abs_activation` = per-neuron mean |activation|, normalized to [0, 1]
- `λ_m = 1.0` because 0.8B models have fewer multimodal neurons, so they deserve stronger protection

### 3.3 Pruning Baselines

| Method | Selection |
|--------|-----------|
| 1. Random pruning | Random neurons globally |
| 2. Layer-matched random | Random, same per-layer count |
| 3. Layer-type-matched random | Random, same per-layer count AND same FA/GDN distribution |
| 4. Magnitude pruning | Lowest mean |activation| |
| 5. Low-activation pruning | Lowest max activation frequency |
| 6. Unknown-score pruning | Highest p_unknown |
| 7. **Unknown + multimodal protection** | prune_score = p_unknown - 1.0*p_multimodal |
| 8. **Unknown + multimodal + outlier protection** | Full prune_score formula |

### 3.4 Outlier Neuron Protection

Some neurons activate broadly and strongly across many tokens without clear semantic pattern. These may be "infrastructure neurons" (distribution calibration, global regulation). If classified as unknown and pruned, damage could be disproportionate.

```python
outlier_score = fraction of samples where normalized activation > T_outlier
```

Neurons with high outlier_score are protected even if p_unknown is high.

### 3.5 Per-Layer Cap

**Do NOT globally prune top-K neurons.** This can devastate specific layers.

Strategy: **equal-ratio per-layer pruning**

```
Each layer prunes the top r% neurons by prune_score within that layer.
```

For a global 20% pruning, each layer prunes its own top 20%. This makes results easier to interpret and prevents layer collapse.

Optional additional cap: no single layer prunes more than 25% even if global target is 20%.

### 3.6 Pruning Ratios

| Ratio | Notes |
|-------|-------|
| 5% | Conservative |
| 10% | Main result |
| 15% | Main result |
| 20% | Aggressive |
| 30% | Stress test |

### 3.7 Recovery Training (Phase 3B, optional)

| Mode | Description |
|------|-------------|
| Zero-shot pruning | Prune → evaluate immediately. Primary results. |
| Light LoRA recovery | 1k-5k image-text instruction samples, 1 epoch, LoRA only. Supplementary. |

**Main conclusion should be based on zero-shot pruning.** Otherwise reviewers will attribute improvements to recovery training, not pruning selection.

### 3.8 Evaluation

Same tasks as Phase 2. Additionally:

| Task | Purpose |
|------|---------|
| Speedup measurement | Wall-clock inference time before/after structural pruning |
| Memory reduction | Peak GPU memory before/after |

### 3.9 Outputs: Phase 3

| Output | Format |
|--------|--------|
| `pruning_masks/` | Per-ratio, per-method: binary mask tensor |
| `pruning_results.jsonl` | Per-method, per-ratio, per-task: score, delta, speedup |
| `fig_pruning_comparison.png` | Line chart: ratio × score for all methods |
| `fig_type_aware_vs_baselines.png` | Bar chart: at fixed ratio, all methods compared |
| `fig_per_layer_pruning.png` | Heatmap: which neurons pruned in each layer |

---

## Code Structure

```
scripts/vulcan/
  neuron_typing/
    collect_ffn_activations.py    # Two-pass activation collection
    score_neuron_types.py         # Compute p_visual/p_text/p_multimodal/p_unknown
    statistical_tests.py          # Blocked permutation test, bootstrap CI
    run_phase1.py                 # End-to-end Phase 1 pipeline
  ablation/
    run_typed_ablation.py         # Phase 2 ablation experiments
    evaluate_tasks.py             # Unified task evaluation
  pruning/
    run_type_aware_pruning.py     # Phase 3 pruning experiments
    compute_pruning_score.py      # prune_score computation
```

### Reuse from Existing Codebase

| Component | Source | Reuse as-is? |
|-----------|--------|:---:|
| Model loading | `collect_cluster_idx.py` | Yes |
| DataLoader + token masks | `collect_cluster_idx.py` multimodal mode | Extend |
| `down_proj` hook | `ActivationAligner` in `activation_align.py` | Yes |
| `find_mlp_layers()` | `modeling.py` | Yes |
| Per-layer intermediate sizes | `VulcanQwen3_5Config` | Yes |

### New Components

| Component | Purpose |
|-----------|---------|
| Per-neuron global max accumulator | Pass 1 |
| Per-sample visual/text activation counter | Pass 2 |
| Per-neuron top-K heap | Top-K sample selection |
| p_type score computation | Final typing |
| Blocked permutation test | FA vs GDN comparison |
| Masked ablation hook | Phase 2 |
| Pruning score + mask generation | Phase 3 |

---

## Experiment Timeline

| Week | Phase | Milestone |
|------|-------|-----------|
| 1 | Pilot | 500-sample pilot, activation scale analysis, threshold calibration |
| 1-2 | Phase 1 | 5k-sample neuron typing, threshold sweep, FA vs GDN comparison |
| 2-3 | Phase 2 | Causal ablation on 3 task categories, same-layer swap |
| 3-4 | Phase 3 | Pruning comparison, zero-shot evaluation |
| 4 | Analysis | Final figures, statistical tests, write-up |

---

## References

1. Deciphering Functions of Neurons in Vision-Language Models (arXiv:2502.18485)
2. Existing Vulcan infrastructure: `collect_cluster_idx.py`, `activation_align.py`, `VulcanQwen3_5ForConditionalGeneration`

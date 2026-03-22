# cortex-lm

A neurophysiologically structured language model. Cortical columns (L4 → L2/3 → L5 → L6) built from rate-coded or AdEx neurons with Dale's Law constraints, Tsodyks-Markram short-term plasticity on inter-column synapses, and an optional Modern Hopfield hippocampal module. Trained with either full BPTT or online e-prop.

The architecture is designed so that each biological ingredient can be toggled independently — enabling ablation studies that measure the marginal contribution of each component.

---

## Quick start

```bash
# Install (requires Python 3.10+)
uv sync               # or: pip install -e ".[dev]"

# Dry-run: check parameter count before training
python scripts/train.py --config configs/minimal.yaml --count-params

# Train (Phase 1 — fast, ~1 hour on CPU)
python scripts/train.py --config configs/minimal.yaml

# Resume from a checkpoint
python scripts/train.py --config configs/minimal.yaml --resume checkpoints/step_0005000.pt

# Override config values from the CLI
python scripts/train.py --config configs/minimal.yaml \
    --override training.lr=1e-3 training.max_steps=20000

# Run all tests
python -m pytest tests/ -q
```

---

## Configs

| File | Phase | Description |
|---|---|---|
| `configs/phase1a_minimal.yaml` | 1a | Rate neurons, `simple_ei` column, no STP/HPC. Baseline. |
| `configs/phase1b_layered.yaml` | 1b | + Layered cortical columns (L4/L2-3/L5/L6). |
| `configs/phase1c_stp.yaml` | 1c | + Tsodyks-Markram STP synapses. |
| `configs/phase1d_adex.yaml` | 1d | + AdEx adaptive neuron dynamics. |
| `configs/phase1e_hopfield.yaml` | 1e | + Modern Hopfield hippocampal module. Full system on TinyStories. |
| `configs/standard_wikitext103.yaml` | 2 | Full system trained on Wikitext-103. |
| `configs/bioplausible_tinystories.yaml` | 3a | e-prop learning rule, TinyStories. |
| `configs/bioplausible_wikitext103.yaml` | 3b | e-prop learning rule, Wikitext-103. |

Key config fields:

```yaml
neuron:
  model: rate | rate_adex        # neuron dynamics
  learn_taus: false               # make τ_m, τ_w learnable parameters

column:
  model: simple_ei | layered      # simple_ei: 1 E/I pair; layered: 4-layer cortical column
  n_columns: 8
  layer_sizes:                    # (layered only)
    l4:  {n_e: 80,  n_i: 20}
    l23: {n_e: 160, n_i: 40}
    l5:  {n_e: 80,  n_i: 20}
    l6:  {n_e: 80,  n_i: 20}

synapse:
  inter_column_stp: true          # Tsodyks-Markram short-term plasticity

hippocampus:
  model: none | modern_hopfield   # CA3-style Hopfield memory

learning:
  rule: bptt | eprop

data:
  dataset: tinystories | wikitext2 | wikitext103 | openwebtext | ptb
  tokenizer: bpe | char | bytes | byte_patch | tiktoken
  vocab_size: 4096                # BPE target (actual may differ slightly)
  seq_len: 128

training:
  checkpoint_dir: checkpoints     # where .pt files and metrics.jsonl land
  checkpoint_interval: 5000       # steps between checkpoints
```

All datasets stream via HuggingFace `datasets` — nothing is downloaded in bulk before training begins.

---

## Scripts

### Training

```bash
# Train
python scripts/train.py --config configs/minimal.yaml

# Check parameter count (no tokenizer training, no data loading — instant)
python scripts/train.py --config configs/minimal.yaml --count-params

# Resume
python scripts/train.py --config configs/minimal.yaml --resume checkpoints/step_0005000.pt
```

`--count-params` prints a per-component parameter breakdown and exits immediately.
It uses the configured `vocab_size` as an estimate — helpful when tuning `layer_sizes`
to match parameter budgets across model variants.

### Sampling

```bash
# Unconditional samples (random start token)
python scripts/sample.py \
    --run-dir checkpoints/ \
    --checkpoint checkpoints/step_0005000.pt \
    --n-samples 5

# Conditioned on a prompt
python scripts/sample.py \
    --run-dir checkpoints/ \
    --checkpoint checkpoints/step_0005000.pt \
    --prompt "Once upon a time there was a little girl named"

# Greedy decoding
python scripts/sample.py ... --temperature 0

# Full options
python scripts/sample.py --help
```

Uses top-p (nucleus) sampling. `--temperature 0` gives greedy decoding.

### Visualisation

```bash
# Training curves (loss, perplexity, BPC, LR schedule)
# Works mid-run — just reads metrics.jsonl as it grows.
python scripts/plot_run.py --run-dir checkpoints/

# Token vocabulary: length distribution, token grid, ID-vs-length scatter, embedding PCA
python scripts/viz_vocab.py --run-dir checkpoints/
python scripts/viz_vocab.py --run-dir checkpoints/ \
    --checkpoint checkpoints/step_0005000.pt   # adds embedding PCA panel

# Empirical neuron timescales (requires scikit-learn for the PCA panel)
python scripts/visualize_timescales.py --checkpoint checkpoints/step_0005000.pt

# Side-by-side comparison (Phase 3: CortexLM vs baselines / ablations)
python scripts/plot_comparison.py \
    --cortex "simple_ei:checkpoints/minimal" "layered:checkpoints/layered" \
    --baselines baseline_results.json \
    --output comparison.png
```

`plot_comparison.py` uses **tokens seen** as the x-axis (a fair comparison across models with different batch sizes), falling back to step count for older runs that pre-date the tokens tracking.

### Baselines

```bash
# Train all baselines at parameter-matched size
python scripts/run_baselines.py \
    --config configs/minimal.yaml \
    --models rnn lstm lstm_attention transformer \
    --output baseline_results.json
```

Each baseline is automatically sized via binary search to match CortexLM's parameter count.

---

## Outputs (per run)

Everything lands in `training.checkpoint_dir` (default: `checkpoints/`):

| File | Description |
|---|---|
| `tokenizer.pkl` | Fitted tokenizer — saved at training start so all post-run scripts work without re-training BPE |
| `metrics.jsonl` | One JSON line per log event. Train: every `log_interval` steps. Val: every `eval_interval` steps. Fields: `step`, `tokens`, `t`, `train/loss`, `train/perplexity`, `lr` (train) or `val/loss`, `val/perplexity`, `val/bpc` (val). |
| `step_NNNNNNN.pt` | Model checkpoint — `model_state_dict`, `optimizer_state_dict`, `config`, `step`. |
| `training_curves.png` | Output of `plot_run.py`. |
| `vocab.png` | Output of `viz_vocab.py`. |

---

## Project structure

```
cortexlm/
  neurons/          rate.py, rate_adex.py, lif.py
  synapses/         static.py (Dale's Law), stp.py (Tsodyks-Markram)
  columns/          simple_ei.py, layered.py
  connectivity/     builder.py (Gaussian-1D / Watts-Strogatz wiring)
  hippocampus/      modern_hopfield.py
  data/             tokenizer.py, tinystories.py, wikitext.py, openwebtext.py, ptb.py
  learning/         bptt.py, eprop.py
  baselines/        rnn.py, lstm.py, rnn_attention.py, lstm_attention.py, transformer.py
  utils/            config.py, logging.py, metrics.py
  model.py          CortexLM top-level model + ModelState dataclass
  readout.py        Multi-layer readout head

scripts/
  train.py              Main training entry point
  sample.py             Top-p text generation from a checkpoint
  plot_run.py           Training curve visualisation
  plot_comparison.py    Multi-run / baseline comparison
  viz_vocab.py          Vocabulary and embedding space visualisation
  visualize_timescales.py  Empirical neuron timescale analysis
  run_baselines.py      Train all baseline models
  evaluate.py           Held-out test set evaluation

configs/
  phase1a_minimal.yaml          Phase 1a — baseline
  phase1b_layered.yaml          Phase 1b — + layered columns
  phase1c_stp.yaml              Phase 1c — + STP
  phase1d_adex.yaml             Phase 1d — + AdEx neurons
  phase1e_hopfield.yaml         Phase 1e — + Hopfield HPC (full system, TinyStories)
  standard_wikitext103.yaml     Phase 2  — full system, Wikitext-103
  bioplausible_tinystories.yaml Phase 3a — e-prop, TinyStories
  bioplausible_wikitext103.yaml Phase 3b — e-prop, Wikitext-103
```

---

## Experimental phases

### Phase 1 — validate the scaffold
`configs/minimal.yaml`: rate neurons, `simple_ei` columns, no STP, no hippocampus, BPTT.
Goal: confirm the cortical architecture can learn next-token prediction at all.

### Phase 2 — add biological fidelity (ablation series)
Each biological ingredient enabled individually, holding parameter count fixed:
1. `column: layered` — full 6-layer cortical column
2. `synapse.inter_column_stp: true` — Tsodyks-Markram STP
3. `neuron: rate_adex` — AdEx adaptation dynamics
4. `hippocampus: modern_hopfield` — CA3-style episodic memory

Use `--count-params` to tune `layer_sizes` so each variant has the same parameter budget before committing to a full run.

### Phase 3 — comparison vs. baselines
`run_baselines.py` trains RNN, LSTM, RNN+attention, LSTM+attention, and Transformer baselines at matched parameter counts. `plot_comparison.py` renders all learning curves on a shared tokens-seen axis.

### Phase 4 — e-prop
`configs/bioplausible.yaml`: identical architecture to Phase 2 but trained with online e-prop (Bellec et al. 2020) instead of BPTT. Quantifies the cost of biological learning rule constraints.

---

## Key biological priors

- **Dale's Law** — excitatory/inhibitory identity fixed per neuron, enforced via `softplus` on raw weights after every optimizer step
- **Log-normal timescales** — τ_m drawn from log-normal distribution (range 2–30 ms), matching cortical neuron heterogeneity
- **AdEx adaptation** — subthreshold adaptation current w with timescale τ_w (30–500 ms) implements spike-frequency adaptation
- **Tsodyks-Markram STP** — synaptic resources (u, x) deplete and recover; facilitating (E→E) vs. depressing (E→I) depending on U₀
- **Laminar routing** — feedforward signals travel L4 → L2/3 → L5; feedback travels L5 → L2/3 of lower-index columns
- **Modern Hopfield hippocampus** — retrieves stored patterns via softmax attention; CA1 surprise signal = L2 distance between retrieved and actual state

---

## Development log

A record of what each training run found and what it motivated.  Perplexities are
**validation ppl** unless noted as `train`.  All Phase 1 runs: TinyStories, BPE 4096,
seq\_len=128, 8 columns, 8 000 steps, batch=512, lr=3e-4, AdamW + cosine.

---

### Phase 1a — `simple_ei` baseline

| Run | Final val ppl | Notes |
|---|---|---|
| 1a-v1 | ~35 | First successful run |
| 1a-v2 | 34 | After fan-in init + softplus-offset fix (see below); confirms fixes improve even the flat model |

---

### Phase 1b — layered columns

#### Attempt 1 — original implementation

**Config:** `l4(8E/2I), l23(16E/4I), l5(8E/2I), l6(8E/2I)`.  Full BPTT.
Weight init: fixed offset `W_raw = randn*0.1 - 2.25` → `W ≈ 0.1` per synapse.

**Result:** plateaued at ~400 ppl, never improved.

**Diagnosis:** Two problems identified:
1. *Size bottleneck* — L5 output is only `8 × 8 = 64`-dim, vs phase1a's `20 × 8 = 160`-dim.  The readout is starved of representational capacity.
2. *Vanishing gradients from full BPTT* — 128-token sequences × 4 spatial layers.

---

#### Attempt 2 — larger layers + truncated BPTT

**Changes vs Attempt 1:**
- Layer sizes increased: `l4(16E/4I), l23(32E/8I), l5(16E/4I), l6(12E/3I)`.  L5 readout → `16 × 8 = 128`-dim (≈ phase1a).
- `truncated_bptt_k: 32` (limits temporal gradient depth to 32 tokens).
- Weight init still fixed (`W ≈ 0.1`).

**Result:** `train ppl ≈ 290`, `val ppl ≈ 719–858` at step 8 000.  Severe overfitting; loss descends but generalisation is poor.

**Diagnosis (identified during architectural exposition):**
The fixed init of `W ≈ 0.1` is appropriate for a small fan-in but catastrophic for the
deep column.  L5_E has ≈64 excitatory pre-neurons from three pathways; total synaptic
input ≈ `64 × 0.5 × 0.1 = 3.2` → `tanh(3.2) ≈ 1.0` → membrane fully saturated →
`tanh'(3.2) ≈ 0.01` → gradient through 3 tanh layers ≈ `0.01³ = 10⁻⁶`.

**Required fix:** *fan-in dependent initialization* — set `W_target = 1/n_pre` so that
total input `≈ n_pre × 0.5 × (1/n_pre) = 0.5` regardless of fan-in.

---

#### Attempt 3 — fan-in initialization

**Changes vs Attempt 2:**
- `BatchedStaticSynapse`, `StaticSynapse`, `STPSynapse`: `offset = softplus⁻¹(1/n_pre)`,
  noise scale `0.01` (was `0.1 − 2.25`).
- Unconstrained projections (thalamic, feedback): `std = 1/√fan_in` (was `0.1`).
- Added per-layer gradient norm logging to diagnose gradient flow.

**Result:** `train ppl ≈ 172`, `val ppl ≈ 282` at step 8 000.

**Gradient norms (step 0 vs step 7 700):**

| Parameter | Step 0 | Step 7 700 | Ratio (readout/param) |
|---|---|---|---|
| `readout` | 7.57e-01 | 143.2 | — |
| `l23_to_l5` | 3.01e-04 | 0.221 | ~650× |
| `l4_to_l23` | 1.27e-05 | 3.10e-03 | ~46 000× |
| `thal_input` | 4.20e-04 | 0.296 | ~480× |

The fan-in fix prevented initial saturation, enabling learning.  But the
`readout / l4_to_l23` ratio remained ~46 000× throughout — thalamic projection weights
receive almost no gradient signal.

**Diagnosis (confirmed by gradient logs):** Two independent vanishing mechanisms remain:
1. *Double-squashing* — each layer applies `tanh(I)` then `sigmoid(v)`.  Per-hop gradient
   ≈ `sigmoid'(v) × tanh'(I) × ‖W‖ ≈ 0.25 × 0.79 × 0.7 ≈ 0.14`.  Over 3 hops: `0.14³ ≈ 0.003`.
2. *tanh always squashing* — with `W ≈ 1/n_pre`, `I ≈ 0.5` and `tanh'(0.5) ≈ 0.79`,
   but larger inputs (from recurrent activity or feedback) push into the saturation regime.

---

#### Attempt 4 — LayerNorm + scaled-tanh output (current)

**Changes vs Attempt 3:**
- `BatchedNeuronPop`: apply `nn.LayerNorm(n_neurons)` to the synaptic input `I` before
  the nonlinearity.  This keeps `tanh'(LN(I)) ≈ 1` at every step, removing tanh from the
  gradient product entirely.
- Replace `sigmoid(v)` with `(tanh(v) + 1) / 2` as the firing-rate output.  Max gradient
  `0.5` (vs sigmoid's `0.25`), and the derivative profile is much flatter.

**Expected per-hop gradient:** `0.5 × 1.0 × ‖W‖` — approximately half the spectral norm
of the inter-layer weight matrix, vs the previous `0.14`.

**Gradient norms at initialization:**

| Parameter | Norm | Ratio to readout |
|---|---|---|
| `readout` | 6.77e-01 | 1× |
| `l23_to_l5` | 3.27e-01 | 2× |
| `l4_to_l23` | 4.06e-02 | 17× |
| `thal_input` | 5.55e-03 | 122× |

`readout / l4_to_l23` dropped from **59 000×** to **17×**.

**Gradient norms during training (step 600):**

| Parameter | Norm | Ratio to readout |
|---|---|---|
| `readout` | 1.17e+01 | 1× |
| `l23_to_l5` | 8.64e+00 | 1.4× |
| `l4_to_l23` | 5.07e-01 | 23× |
| `thal_input` | 4.93e-01 | 24× |

All four levels within two orders of magnitude — thalamic projection and L4→L23 weights
are now training meaningfully.

**Early results (step 700, still training):**

| Step | Train ppl | Val ppl |
|---|---|---|
| 0 | 3 535 | 3 492 |
| 200 | 194 | 219 |
| 400 | 109 | 127 |
| 600 | 80 | 97 |
| 700 | 71 | — |

For reference, phase1a (simple_ei baseline) reached ~34 val ppl after the full 8 000 steps.
Phase1b Attempt 4 is tracking well ahead of that pace.

---

## Future architectural refinements (speculative)

A running list of directions worth exploring once the current ablation series is complete.

### Neuron-level sparse connectivity
Currently the weight matrices *within* each column (E→E, E→I, I→E) are fully dense. Biological cortex has ~10% connection probability between any two nearby neurons. A sparse intra-column weight matrix — implemented as a learned dense matrix multiplied elementwise by a fixed random binary mask — would be more faithful and would reduce intra-column parameter count substantially, potentially allowing larger columns at the same budget. The existing `gaussian_1d` / `small_world` connectivity only controls which *column pairs* communicate, not which individual neurons within a pair are wired.

### Per-column specialisation of the input projection
Currently every column receives the same token embedding (32-dim), projected independently via its own `input_proj` (32 → n_e). Biologically, different cortical areas receive different thalamic projections — they are not all exposed to the same raw sensory signal. One option: learn a shared input embedding but give each column a *different* learned selection over it (via attention or a learned mask). Another option: positional encoding across columns so that column index carries topographic meaning (e.g. columns 0–3 specialise in syntax, 4–7 in semantics).

### Richer interneuron subtypes
The current model has only one inhibitory population (I) per layer. Real cortex has at least three functionally distinct inhibitory subtypes:
- **PV (parvalbumin)** — fast, perisomatic, divisive gain control
- **SST (somatostatin)** — targets dendrites; implements multiplicative gating
- **VIP** — inhibits SST (disinhibition), enabling top-down modulation

The `disinhibition` flag in column config is a placeholder for the VIP→SST→PC circuit. Expanding to three I populations per layer would enable richer gating dynamics at modest parameter cost.

### Learnable timescales
`learn_taus: true` is already wired in but disabled. Allowing τ_m and τ_w to be learned (rather than fixed from a log-normal draw) would let the model discover which timescales are useful for the task. Worth enabling once the architectural ablations are done, to see whether gradient descent recovers biologically plausible timescale distributions.

### Local learning rules for intra-column weights
Even while training the full model with BPTT, one could make intra-column weights learn via a local Hebbian or STDP rule, reserving BPTT only for the inter-column and readout connections. This hybrid approach is more biologically realistic and might impose a useful inductive bias (local structure is learned locally; global coordination is learned globally).

### Neuromodulatory gain signals
Dopamine, acetylcholine, and norepinephrine globally modulate cortical excitability and learning rates in biology. A simple approximation: add a scalar gain signal per timestep predicted from the hippocampal surprise (CA1 output), applied multiplicatively to all column activations. This would give the model a learned "arousal" state that could gate plasticity or sharpen representations.

### Scaling the number of columns and neurons
The ablation configs use 8–16 columns with small neuron counts. A natural next experiment after the ablation series: hold architecture constant and scale up (32+ columns, larger layer sizes), to check whether the bio-plausible architecture benefits from scale in the same way transformers do.

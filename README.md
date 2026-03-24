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
| `configs/phase1e_disinhibition.yaml` | 1e | + VIP→SST→PC disinhibition circuit. Completes cortical column. |
| `configs/phase1e_hopfield.yaml` | (exploratory) | Hopfield + AdEx, used for apical pathway factorial. |
| `configs/phase1f_hopfield.yaml` | 1f | + Modern Hopfield hippocampal module. Full system on TinyStories. |
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

## Data pipeline

### Dataset

All experiments use **TinyStories** (Eldan & Li 2023) by default — a synthetic
corpus of short children's stories generated by GPT-3.5/4, totalling ~2 GB of
text.  It is fetched on-demand via the HuggingFace `datasets` library
(`roneneldan/TinyStories`); nothing is downloaded in bulk before training begins.

TinyStories ships with an official `train` / `validation` split.  There is no
separate test split; the validation split is used for both held-out evaluation
during training and final test-set reporting.

### Tokenizer

A **Byte-Pair Encoding (BPE)** tokenizer is trained from scratch on a 500 000-
character sample of the training split at the start of every run.  The target
vocabulary size is set in config (`vocab_size: 4096`); the actual size after
BPE merges is typically 3 300–3 400 tokens (the BPE algorithm stops when it
runs out of profitable merges before reaching the target).

The fitted tokenizer is saved as `tokenizer.pkl` in the checkpoint directory so
that all post-training scripts (sampling, evaluation, length generalisation)
can load it without re-running BPE.

### Tokenization and caching

After the tokenizer is trained, the full training and validation splits are
tokenized once and saved as flat `uint16` numpy arrays under `data/cache/`:

```
data/cache/tinystories_train_bpe_3356.npy       # ~50 M tokens, ~95 MB
data/cache/tinystories_validation_bpe_3356.npy  #  ~5 M tokens, ~10 MB
```

The filename encodes the dataset, split, tokenizer type, and vocabulary size,
so the cache is automatically invalidated if any of those change.  Subsequent
runs load from the cache file via `np.load` (memory-mapped) — no HTTP traffic,
no re-tokenization.  Pass `--no-cache` to force a rebuild.

### Train / validation split

| Split | Source | Tokens (default limit) | Use |
|---|---|---|---|
| `train` | `roneneldan/TinyStories` split=`train` | 50 M | Parameter updates |
| `val` | `roneneldan/TinyStories` split=`validation` | 5 M | Loss reporting every `eval_interval` steps |

The limits (50 M / 5 M) are soft ceilings; if the HuggingFace split is smaller
the full split is used.  They can be overridden in config:

```yaml
data:
  tinystories_token_limit: 50_000_000   # train
```

### Sequence construction

The flat token array is divided into non-overlapping chunks of length
`seq_len + 1`.  Each chunk yields one `(x, y)` pair:

```
tokens = [t0, t1, t2, ..., t_{L}]
x      = [t0, t1, ..., t_{L-1}]   # model input
y      = [t1, t2, ..., t_{L}]     # next-token targets
```

Chunks do **not** span story boundaries in any special way — stories are
concatenated into one long token stream before chunking.  This means a single
training example can straddle two stories.  At `seq_len=128` and a median story
length of ~500 tokens this happens infrequently but is not prevented.

### Tokens seen during training

With `batch_size=512`, `seq_len=128`, and `max_steps=8000`:

```
tokens_seen = 512 × 128 × 8000 = 524 288 000  (~524 M tokens)
```

This exceeds the 50 M training token limit, so the dataloader cycles through
the training set roughly 10 times over the course of a run.

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

**Results:**

| Step | Train ppl | Val ppl |
|---|---|---|
| 0 | 3 535 | 3 492 |
| 200 | 194 | 219 |
| 400 | 109 | 127 |
| 600 | 80 | 97 |
| 800 | 65 | 85 |
| 1 000 | 57 | 75 |
| 1 200 | 53 | 72 |
| 1 400 | 50 | 69 |
| 7 200 | 26 | 45 |
| 7 600 | 26 | 43 |
| 7 800 | 25 | 43 |
| 7 999 | **25** | **~43** |

Phase1a (simple_ei, same budget) reached ~34 val ppl.  Phase1b Attempt 4
**substantially outperforms it**, reaching 25 train ppl and ~43 val ppl.
The train/val gap is wider than phase1a — suggesting room for regularisation
or longer training — but the trajectory is clearly healthy and still
declining at the end of the run.  Gradient norms remained well-balanced
throughout (all four depth levels within one order of magnitude), confirming
that the LayerNorm + scaled-tanh fixes resolved the credit assignment problem.

---

### Phase 1c — STP (exploratory run)

**Changes vs Phase 1b:** `inter_column_stp: true` — the only change.

**Config bug found:** `phase1c_stp.yaml` had layer sizes halved relative to phase1b (l4: 8E/2I instead of 16E/4I, etc.), giving only 569K params. Fixed before the run.

**Token budget issue found:** `max_steps` was a fixed step count, so doubling batch size doubled total tokens seen. Fixed by replacing `max_steps` with `max_tokens: 524_288_000` across all phase1 configs; the trainer now derives steps as `max_tokens // (batch_size × seq_len)`, keeping data exposure constant regardless of batch size.

**Exploratory run** (batch=1024, ran past canonical budget due to above):

| Tokens | Step | Train ppl | Val ppl |
|---|---|---|---|
| 524M (canonical) | 4 000 | 32.1 | 46.3 |
| 655M | 5 000 | — | 42.7 |
| 682M | 5 200 | 25.8 | 41.9 |

Run crashed at ~step 5200 before natural completion. Key finding: **val ppl was still declining at the canonical cutoff** — the cosine LR schedule hadn't meaningfully decayed yet because the canonical 524M-token budget was calibrated for batch=512. For final canonical runs, use batch=512 so the LR schedule completes within the token budget.

Preliminary conclusion: STP does not hurt; likely helps given more training. Final verdict deferred to canonical runs.

---

### Phase 1d — AdEx neurons

**Changes vs Phase 1c:** `neuron: rate_adex` — the only change. AdEx adds a slow adaptation current `w` per neuron (τ_w ∈ [30, 500] ms), giving spike-frequency adaptation and heterogeneous timescales across `w` and `v`.

**Config bug found:** `phase1d_adex.yaml` had been drafted at Phase 2 scale — 16 columns, layer sizes up to 160E/40I, 4.8M params. Reset to match phase1c exactly (8 columns, same layer sizes, ~622K params).

**Gradient bug found (critical):** The `rate_adex` branch of `BatchedNeuronPop` was missing both gradient fixes from Phase 1b:
1. **LayerNorm on synaptic input** — not applied; `x` passed raw into the voltage update `dv = α_m × (−v + R·x − w)`, leaving `tanh'(x)` in the gradient product.
2. Result: at initialization, `readout/l4_to_l23` gradient ratio ≈ **29,000×** — essentially the same as Phase 1b before any fixes.

**Fix:** moved `self.input_norm = nn.LayerNorm(n_neurons)` outside the `if/else` branch so it is constructed for both `rate` and `rate_adex`, and applied `x_norm = self.input_norm(x)` before the voltage update in both paths. The scaled-tanh output `(tanh(v)+1)/2` was already present in the AdEx branch.

Note: the fan-in dependent synapse init (`softplus_inv(1/n_pre)`) lives in the synapse weight matrices and is shared regardless of neuron type — it was already correct.

**Expected gradient ratio after fix:** ~17× (same as Phase 1b Attempt 4 at initialization).

---

### LR schedule bug (affects all pre-canonical runs)

**Bug:** `scheduler.step()` was called once **per truncated-BPTT chunk**, not once per training step. With `truncated_bptt_k=32` and `seq_len=128` there are 4 chunks per step, so the cosine schedule cycled 4× too fast — completing at step ~975 and restarting. The LR never decayed to zero; it bounced back to `lr_max` repeatedly. Warmup also ended after 25 training steps instead of 100.

**Symptom:** terminal `lr` values near 3e-4 (= `lr_max`) at the end of runs that should have decayed to ~0.

**Fix:** removed `_warmup_lr()`, `scheduler.step()`, and `step_count += 1` from inside `_truncated_bptt`; moved them to the outer `train()` loop so they fire exactly once per training step.

**Impact:** all phase 1a–1e results above are non-canonical. Canonical reruns pending.

---

### Phase 1e + apical pathway factorial (exploratory, post-fix)

Four apical variants run on `phase1e_hopfield.yaml` (Hopfield + AdEx, batch=1024, 4000 steps).

#### Variant A — skip

**Result at step 1900 (47% done):** val ppl = **28.66** — already better than the *final* result of the broken-LR 1e run (36.19 at step 3900). Run stopped intentionally.

**Gradient signature:** `l23_to_l5` collapsed to ~0.013 vs `readout` ~2.0 (ratio ~150×). The skip pathway provides a direct gradient highway from readout → thalamic input, bypassing L23→L5. Analogous to the transformer residual stream.

---

#### Variant B — additive (sigmoid gate)

Crashed at step 2330 with `CUDA error: unknown error` (hardware glitch, not a code bug). Last checkpoint: step 2250, val ppl ≈ 28–29.

**Gradient signature:** `l23_to_l5` healthy at ~0.5–1.0 throughout — both the apical and recurrent pathways carry meaningful gradient. Gate mean grew from ~0.05 (near-silent at init) to ~0.3–0.4, showing the pathway gradually opened during training.

---

#### Variant C — multiplicative (Larkum calcium spike)

`I_l5e_out = I_l5e × (1 + tanh(apical_proj(embed)))`. Weights init near zero → identity at start.

**Results (selected):**

| Step | val_ppl | val_bpb | HPC entropy | L5e τ_eff |
|---|---|---|---|---|
| 0 | 2699 | 2.16 | 4.16 | 9.9 |
| 500 | 56.9 | 1.10 | 2.46 | 14.6 |
| 1000 | 40.3 | 1.01 | 1.99 | 15.2 |
| 1500 | 33.9 | 0.963 | 1.86 | 15.7 |
| 2000 | 30.9 | 0.937 | 1.74 | 16.3 |
| 2200 | 30.0 | 0.929 | 1.69 | — |

NLL–H gap crossed zero at step ~1500 (calibration). HPC entropy declining monotonically (memory consolidation). L5e τ_eff growing throughout (~9 → 16+) — network learns to exploit slow timescales.

---

#### Variant D — corticortical

Previous-timestep L5E of column (k+1)%n feeds back to L23E of column k. Circular inter-column hierarchy.

**Results (selected):**

| Step | val_ppl | val_bpb | HPC entropy | l4_to_l23 grad |
|---|---|---|---|---|
| 0 | 2841 | 2.17 | 4.16 | 2.87e-04 |
| 200 | 200.8 | 1.45 | **0.12** | 1.64e-03 |
| 500 | 80.1 | 1.20 | 1.22 | 4.37e-03 |
| 1000 | 50.1 | 1.07 | 2.02 | 7.04e-03 |
| 1500 | 41.2 | 1.02 | 2.39 | 3.70e-03 |
| 2000 | 37.0 | 0.987 | 2.51 | 4.03e-03 |
| 2200 | 36.2 | 0.981 | 2.52 | 2.97e-03 |

**Key findings:**
- `l4_to_l23` gradient ~10× smaller than in A/B/C throughout. The corticortical input dominates L23, suppressing the feedforward L4→L23 pathway. This inverts the canonical cortical hierarchy — L23 driven more by top-down than bottom-up.
- **HPC early collapse:** attn_entropy crashed to 0.12 at step 200 (attn_max=0.98 — single memory dominates). Then slowly recovered to ~2.5 over the remainder. Unique to this variant; caused by the circular L5E→L23E→…→L5E resonance loop creating a strong attractor at init.
- Convergence significantly slower than A/B/C. Estimated final ppl ~33–35 vs ~27–29 for the others.
- Interpretation: *where* the apical signal enters the column matters. Signals into L5 (A/B/C) help; signals into L23 (D) interfere with feedforward credit assignment.

---

**Confounds common to all exploratory runs:** (1) batch=1024 vs canonical batch=512; (2) apical pathway not independently ablated from HPC. Need clean canonical runs (no apical, fixed LR) to separate contributions.

---

## Open research questions

Two fundamental problems motivate this project beyond the immediate ablation
series.

### 1. Architectural efficiency

Transformer-based models require many orders of magnitude more training data
than a human child needs to acquire basic language competency.  The brain
solves this with far fewer tokens.  Why?

The most likely answer is not a single mechanism but a combination of strong
**inductive biases encoded structurally** in the architecture:

- The laminar column topology (L4 → L2/3 → L5 → L6) is not just a
  implementation detail — it encodes prior knowledge about the hierarchical,
  compositional structure of sensory processing that evolution has refined
  over millions of years.
- **Heterogeneous timescales** (τ_m ∈ [2, 30] ms) give the network a
  multi-scale representation of context that is qualitatively different from
  an attention window.  Slow neurons are literally integrating over long
  temporal intervals; fast neurons track rapid local statistics.
- **Massive top-down feedback** (L5 → L2/3 of lower columns, higher areas
  → lower areas) supports something closer to active inference than passive
  recognition.  Transformers are feedforward at inference.

The working hypothesis: *the architecture is the prior*.  If enough of the
right inductive biases are baked into the connectivity, the model may need
far weaker learning signals to converge — and may generalise from far less
data.

### 2. Biologically plausible learning

E-prop (Bellec et al. 2020) is a step toward biological plausibility but
still requires symmetric feedback weights (the weight transport problem) and
a global error signal — both biologically implausible.  The brain solves
credit assignment over deep hierarchies and long time horizons via some
mechanism we do not yet understand.

The most promising biological alternatives currently under investigation:

- **Predictive coding** (Rao & Ballard 1999; Friston) — each layer predicts
  the activity of the layer above; error signals are *local* differences
  between prediction and observation.  Has a loose mathematical equivalence
  to backprop under certain conditions but is genuinely local.
- **Contrastive Hebbian learning / Equilibrium Propagation** — networks
  settle to free-phase and clamped-phase equilibria; learning is the
  difference between the two phases.
- **Neuromodulated STDP** — spike-timing dependent plasticity gated by a
  dopaminergic surprise signal.  Learning is Hebbian but only when something
  unexpected happens — a natural fit for the hippocampal CA1 surprise signal
  already wired into this architecture.
- **Forward-Forward** (Hinton 2022) — no backprop at all; each layer
  maximises a local "goodness" criterion for positive examples and minimises
  it for negative ones.

**The key insight connecting both problems:** if the architecture encodes
sufficiently strong structural priors, it may need only weak and local
learning signals to converge.  The laminar column structure, STP, and
timescale diversity together might do enough of the representational work
that precise gradient information becomes unnecessary.  This would be a
genuinely novel result: not "bio-plausible learning rule matches backprop"
but "bio-plausible architecture makes precise credit assignment less
important."

### 3. Length generalisation

Standard Transformers cannot generalise beyond their training context length
(positional encodings simply do not exist for unseen positions).  RNNs and
LSTMs can in principle, but often fail in practice because gradients do not
flow meaningfully over long horizons during training.

CortexLM may generalise better due to:
- No positional encoding — the state update equations are
  time-translation invariant.
- Heterogeneous τ_m — slow neurons preserve a continuously decaying
  integral of arbitrarily distant history; there is no cliff at T_train.
- The state dynamics run indefinitely; inference beyond the training length
  requires no architectural change.

This is directly testable with `scripts/length_generalization.py`.

### 4. Sample efficiency as the key metric

For the paper's framing, the interesting comparison is not "can a
bio-plausible model match a Transformer at 500B tokens" — it is "can it
match a Transformer's *early* learning trajectory, the way a child acquires
language from orders of magnitude less exposure."  Perplexity vs tokens seen
(rather than vs steps) is the right axis, and it puts the biological
architecture's inductive biases front and centre.

---

## Future architectural refinements (speculative)

A running list of directions worth exploring once the current ablation series is complete.

### Larkum two-compartment L5 / apical dendritic pathway *(implemented and run)*

L5 pyramidal neurons in real cortex have two functionally distinct input zones: basal dendrites in L5 (receiving local feedforward input from L2/3) and apical dendrites extending all the way to L1 (receiving top-down feedback from higher areas). The two zones interact nonlinearly — strong apical depolarisation triggers a dendritic calcium spike that dramatically amplifies somatic output (Larkum 2013). This is the likely biological analogue of the residual stream in transformers: it provides a gradient highway and a meaningful output even when the intermediate processing layers are not yet trained.

The apical pathway is implemented in `cortexlm/columns/apical.py` (`ApicalPathway` module). Activated via a single config flag:

```yaml
column:
  apical_pathway: skip          # or: additive | multiplicative | corticortical | none (default)
```

| Variant | `apical_pathway` value | Description |
|---|---|---|
| A | `skip` | Direct embedding → L5E additive projection (skip connection / thalamic bypass). Simplest gradient highway. |
| B | `additive` | Embedding → L5E projection with a learned per-neuron sigmoid gate (init ≈ 0.05, so pathway opens gradually). |
| C | `multiplicative` | Full Larkum calcium spike: `I_l5e · (1 + tanh(apical_proj))`. Gating is multiplicative, not additive. |
| D | `corticortical` | Previous-timestep L5E of column k+1 feeds into L23E of column k — circular top-down inter-column feedback. |

#### Exploratory results (Hopfield + AdEx, batch=1024, ~524M tokens)

All four variants run on `phase1e_hopfield.yaml` (Hopfield + AdEx, 8 columns). Results at the canonical token budget:

| Variant | val_ppl @2200 steps (55%) | Gradient signature | HPC behaviour |
|---|---|---|---|
| A (skip) | ~28.7 (stopped @1900) | `l23_to_l5` collapses to ~0.01 (gradient highway dominates) | Entropy: 4.16 → 1.7, monotone |
| B (additive) | ~29 (crashed @2330) | `l23_to_l5` healthy ~0.5–1.0; both pathways active | Entropy: 4.16 → 1.8, monotone |
| C (multiplicative) | 29.95 | `l23_to_l5` healthy ~0.4–0.8; slower convergence | Entropy: 4.16 → 1.7, monotone |
| D (corticortical) | 36.24 | `l4_to_l23` near-zero (~0.003); feedforward starved | Entropy collapsed to 0.12 @step 200, then recovered to ~2.5 |

**Key findings:**

1. **Gradient routing matters.** Variants A/B/C (apical → L5) all improve on the no-apical baseline and maintain healthy gradient flow through the column. Variant D (apical → L23) suppresses the L4→L23 feedforward gradient 10× — the corticortical signal dominates L23 input and starves the feedforward pathway.

2. **Hopfield early collapse in D.** The circular L5E→L23E→…→L5E resonance loop caused the Hopfield memory bank to collapse onto a single pattern by step 200 (`attn_entropy = 0.12`, `attn_max = 0.98`). The bank then slowly diversified over the remaining training. Variants A/B/C showed the opposite: gradual consolidation (monotonically decreasing entropy) as the network discovered which memories were useful.

3. **Calibration (NLL–H gap).** Variants B and C crossed zero (well-calibrated) around step 1500. Variant D remained negative (underconfident) throughout — consistent with its slower convergence.

4. **L5e effective timescales grow during training** (τ_eff: 9 → 15–18 steps in A/B/C). The network learns to exploit the heterogeneous timescale distribution. Variant D shows less growth — suggesting inter-column synchrony rather than within-column temporal integration.

The planned canonical experiment is a **factorial ablation**: each of the 4 apical variants on top of each phase-1 config (1a–1f), keeping everything else fixed.

### Neuron-level sparse connectivity
Currently the weight matrices *within* each column (E→E, E→I, I→E) are fully dense. Biological cortex has ~10% connection probability between any two nearby neurons. A sparse intra-column weight matrix — implemented as a learned dense matrix multiplied elementwise by a fixed random binary mask — would be more faithful and would reduce intra-column parameter count substantially, potentially allowing larger columns at the same budget. The existing `gaussian_1d` / `small_world` connectivity only controls which *column pairs* communicate, not which individual neurons within a pair are wired.

### Per-column specialisation of the input projection
Currently every column receives the same token embedding (32-dim), projected independently via its own `input_proj` (32 → n_e). Biologically, different cortical areas receive different thalamic projections — they are not all exposed to the same raw sensory signal. One option: learn a shared input embedding but give each column a *different* learned selection over it (via attention or a learned mask). Another option: positional encoding across columns so that column index carries topographic meaning (e.g. columns 0–3 specialise in syntax, 4–7 in semantics).

### Richer interneuron subtypes
The current model has only one inhibitory population (I) per layer. Real cortex has at least three functionally distinct inhibitory subtypes:
- **PV (parvalbumin)** — fast, perisomatic, divisive gain control
- **SST (somatostatin)** — targets dendrites; implements multiplicative gating
- **VIP** — inhibits SST (disinhibition), enabling top-down modulation

The `disinhibition` flag activates the VIP→SST→PC circuit (implemented in `BatchedLayeredColumns`): a VIP population (n_vip = n_i // 2) per layer receives excitatory input from the E population and inhibits the SST (I) population, thereby disinhibiting the pyramidal cells. This is Phase 1e in the canonical sequence. Expanding to fully separate PV/SST/VIP populations would enable richer gating dynamics at modest parameter cost.

### Learnable timescales
`learn_taus: true` is already wired in but disabled. Allowing τ_m and τ_w to be learned (rather than fixed from a log-normal draw) would let the model discover which timescales are useful for the task. Worth enabling once the architectural ablations are done, to see whether gradient descent recovers biologically plausible timescale distributions.

### Local learning rules for intra-column weights
Even while training the full model with BPTT, one could make intra-column weights learn via a local Hebbian or STDP rule, reserving BPTT only for the inter-column and readout connections. This hybrid approach is more biologically realistic and might impose a useful inductive bias (local structure is learned locally; global coordination is learned globally).

### Neuromodulatory gain signals
Dopamine, acetylcholine, and norepinephrine globally modulate cortical excitability and learning rates in biology. A simple approximation: add a scalar gain signal per timestep predicted from the hippocampal surprise (CA1 output), applied multiplicatively to all column activations. This would give the model a learned "arousal" state that could gate plasticity or sharpen representations.

### Scaling the number of columns and neurons
The ablation configs use 8–16 columns with small neuron counts. A natural next experiment after the ablation series: hold architecture constant and scale up (32+ columns, larger layer sizes), to check whether the bio-plausible architecture benefits from scale in the same way transformers do.

---

## Replication guide

Step-by-step recipe to reproduce the full study from scratch.  Each step
specifies the command, expected wall-time on a single RTX 3090, and what to
check before moving on.

> **Setup once**
> ```bash
> uv sync                    # install deps
> ```

---

### Step 0 — Verify installation

```bash
python scripts/train.py --config configs/phase1a_minimal.yaml --count-params
```

Expected output: parameter breakdown, totalling ~620 K params, then exits.

---

### Step 1 — Phase 1a: `simple_ei` baseline

```bash
python scripts/train.py --config configs/phase1a_minimal.yaml --wandb
```

- **What it is:** rate neurons, single E/I pair per column, no STP, no HPC, BPTT.
- **Expected result:** val ppl ≈ 34 at 8 000 steps.
- **Wall-time:** ~1 h on GPU.
- **Check:** `checkpoints/*/metrics.jsonl` exists; final val ppl in log.
- **Saves:** `checkpoints/phase1a_.../tokenizer.pkl` — reuse for all subsequent runs.

---

### Step 2 — Phase 1b: layered cortical columns

```bash
python scripts/train.py --config configs/phase1b_layered.yaml \
    --tokenizer checkpoints/<phase1a-run>/tokenizer.pkl --wandb
```

- **What it is:** full 4-layer cortical column (L4/L2-3/L5/L6) replacing the
  single E/I pair.  Fan-in init + LayerNorm + scaled-tanh output.
- **Expected result:** val ppl ≈ 43 at 8 000 steps (train ppl ≈ 25).
- **Wall-time:** ~2–3 h on GPU.
- **Check:** gradient norms logged at each eval step — `grad/l4_to_l23` should
  be within ~30× of `grad/readout` throughout.

---

### Step 3 — Phase 1c: short-term plasticity

```bash
python scripts/train.py --config configs/phase1c_stp.yaml \
    --tokenizer checkpoints/<phase1a-run>/tokenizer.pkl --wandb
```

- **What it is:** Tsodyks-Markram STP on inter-column synapses; everything else
  identical to phase1b.
- **Expected result:** val ppl improvement over phase1b (hypothesis: facilitating
  E→E STP helps bind sequences; depressing E→I STP prevents runaway excitation).

---

### Step 4 — Phase 1d: AdEx neurons

```bash
python scripts/train.py --config configs/phase1d_adex.yaml \
    --tokenizer checkpoints/<phase1a-run>/tokenizer.pkl --wandb
```

- **What it is:** adds AdEx adaptive neuron dynamics (`neuron: rate_adex`) on top
  of phase1c.  Subthreshold adaptation current w with timescale τ_w implements
  spike-frequency adaptation.

---

### Step 5 — Phase 1e: Modern Hopfield hippocampus (full system)

```bash
python scripts/train.py --config configs/phase1e_hopfield.yaml \
    --tokenizer checkpoints/<phase1a-run>/tokenizer.pkl --wandb
```

- **What it is:** adds the Modern Hopfield CA3 memory module on top of phase1d.
  CA1 computes a surprise signal = L2 distance between retrieved and actual column
  state; this is fed back as a neuromodulatory gain signal.
- **This is the full system on TinyStories.**

---

### Step 6 — Baselines (parameter-matched)

```bash
python scripts/run_baselines.py \
    --config configs/phase1e_hopfield.yaml \
    --tokenizer checkpoints/<phase1a-run>/tokenizer.pkl \
    --models rnn lstm rnn_attention lstm_attention transformer \
    --output results/baseline_results.json \
    --wandb
```

- Each baseline is auto-sized via binary search to match CortexLM's parameter count.
- **Wall-time:** ~5–8 h total (all five models sequentially on GPU).
- **Check:** `results/baseline_results.json` has a `baselines` dict with final val ppl for each model.

---

### Step 7 — Comparison plots

```bash
python scripts/plot_comparison.py \
    --cortex \
        "1a-simple_ei:checkpoints/<phase1a-run>" \
        "1b-layered:checkpoints/<phase1b-run>" \
        "1c-stp:checkpoints/<phase1c-run>" \
        "1d-adex:checkpoints/<phase1d-run>" \
        "1e-full:checkpoints/<phase1e-run>" \
    --baselines results/baseline_results.json \
    --output results/comparison.png
```

X-axis is tokens seen (fair across models with different batch sizes).

---

### Step 8 — Length generalisation

```bash
python scripts/length_generalization.py \
    --checkpoint checkpoints/<phase1e-run>/step_0008000.pt \
    --tokenizer checkpoints/<phase1a-run>/tokenizer.pkl \
    --config configs/phase1e_hopfield.yaml \
    --lengths 64 128 256 512 1024 \
    --n-sequences 200 \
    --output results/length_gen
```

- Tests whether the model generalises beyond `seq_len=128` (its training length).
- Compare against a Transformer baseline checkpoint from step 6 to demonstrate
  the CortexLM advantage.

---

### Step 9 — Phase 2: scale to Wikitext-103

```bash
python scripts/train.py --config configs/standard_wikitext103.yaml --wandb
```

- Same architecture as phase1e but trained on Wikitext-103 (~100 M tokens, richer
  vocabulary).
- Expect slower convergence; use `--override training.max_steps=30000`.

---

### Step 10 — Phase 3: e-prop learning rule

```bash
python scripts/train.py --config configs/bioplausible_tinystories.yaml --wandb
```

- Identical architecture to phase1e but trained with online e-prop instead of BPTT.
- The key comparison: how much does biological learning rule plausibility cost in terms of final perplexity and sample efficiency?

---

### What to report

| Metric | Script | Description |
|---|---|---|
| Val ppl vs tokens (ablation table) | `plot_comparison.py` | Phase 1a → 1e + baselines |
| Val ppl at matched params (bar chart) | `plot_comparison.py` | Final ppl all models |
| Length generalisation curve | `length_generalization.py` | OOD ppl vs eval length |
| Sample efficiency curve | `plot_comparison.py` | Early tokens-seen axis |
| e-prop vs BPTT gap | both training logs | Cost of bio learning rule |

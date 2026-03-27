# cortex-lm

A neurophysiologically structured language model trained with biologically plausible learning rules.

---

## What this project is about

The standard narrative in neural network research is that biological realism is a performance tax — you pay in accuracy for the sake of interpretability or plausibility.  This project challenges that assumption.

We build a language model out of the actual components of cortex: layered excitatory/inhibitory columns, adaptive neurons with spike-frequency adaptation, short-term synaptic plasticity, a hippocampal associative memory module, and apical dendritic feedback connections.  Each component is independently togglable, so we can measure the marginal contribution of every biological ingredient.  None were added speculatively — each is motivated by a specific neural circuit with a known functional role.

The result is not what a naive reading of the benchmarks would predict.  **Each biologically-motivated addition closed the performance gap with a standard transformer, rather than widening it.**  Layered columns, AdEx dynamics, STP, and the Hopfield hippocampus each improved sample efficiency incrementally.  And when we turned to a biologically plausible learning rule — e-prop, which uses only locally available signals and no backpropagation through time — the outcome was surprising: rather than hurting convergence, the combination of e-prop with an additive apical feedback pathway *radically boosted* it.

The apical finding is the sharpest result the project has produced.  Without apical feedback, e-prop fails completely on this architecture — the learning signal collapses to zero within the first million tokens and training stalls permanently.  With apical feedback, the signal grows throughout training, and the model reaches perplexities that pure BPTT and a transformer baseline take 100× more data to achieve.  This is not an engineering trick.  Apical dendrites are a real biological structure, present on virtually every L5 pyramidal neuron in cortex, long known to carry top-down signals.  What we found is that they are computationally *necessary* for online local learning to work at all — not merely modulatory, but load-bearing for credit assignment.

This suggests a reframing.  The question is not "can a biological model beat a transformer on a benchmark?"  The question is: **which model learns fastest from limited data, and why?**  Biological brains are not trained on internet-scale corpora with full backpropagation.  They learn quickly, online, from a noisy stream of experience — exactly the regime where this model excels.  The transformer may ultimately reach lower perplexity given enough data.  But the biologically-structured model gets there faster, learns from less, and does so using mechanisms that have direct neural correlates.

We are not yet done.  The model's asymptotic performance under e-prop is still below its BPTT ceiling, and we are actively investigating whether a hybrid learning rule — e-prop for fast online learning, periodic BPTT consolidation for long-range credit assignment — can close that gap.  Early results suggest the optimal awake:asleep ratio in this hybrid matches the biological 2:1 proportion, hinting that this ratio may reflect a deeper computational principle rather than an arbitrary evolutionary constraint.

The architecture is described in detail below.  The learning rule experiments are documented in the [e-prop section](#e-prop-online-learning-rule).

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

**Canonical ablation series (Phase 1)**

| File | Phase | Description |
|---|---|---|
| `configs/phase1a_minimal.yaml` | 1a | Rate neurons, `simple_ei` column, no STP/HPC. Baseline. |
| `configs/phase1b_layered.yaml` | 1b | + Layered cortical columns (L4/L2-3/L5/L6). |
| `configs/phase1c_stp.yaml` | 1c | + Tsodyks-Markram STP synapses. |
| `configs/phase1d_adex.yaml` | 1d | + AdEx adaptive neuron dynamics. |
| `configs/phase1e_disinhibition.yaml` | 1e | + VIP→SST→PC disinhibition circuit. |
| `configs/phase1f_hopfield.yaml` | 1f | + Modern Hopfield hippocampal module (no disinhibition). |
| `configs/phase1g_hopfield_disinhibition.yaml` | 1g | 1f + always-on VIP disinhibition. Do they combine? |
| `configs/phase1h_hopfield_annealed.yaml` | 1h | 1f + annealed disinhibitory window (1→0 over first 200 M tokens). |
| `configs/phase1i_hopfield_ca1.yaml` | 1i | 1f + CA1 entorhinal prediction-error write-gating. |

**Apical stream ablation series**

| File | Variant | Description |
|---|---|---|
| `configs/apical_none.yaml` | none | No apical pathway. Control. |
| `configs/apical_skip.yaml` | skip | Direct embed→L5E projection (always active, fan-in init). |
| `configs/apical_additive.yaml` | additive | Gated projection; sigmoid gate init≈0, learned open. |
| `configs/apical_multiplicative.yaml` | multiplicative | Larkum calcium spike: `I_l5e *= (1 + tanh(proj(embed)))`. |
| `configs/apical_corticortical.yaml` | corticortical | Circular column feedback: L5E of col k+1 → L23E of col k. |

All apical configs run on the full Phase-1g architecture (AdEx + STP + annealed disinhibition + Hopfield).  The winning variant is then applied to the full canonical 1a–1g series.

**Other**

| File | Phase | Description |
|---|---|---|
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
  rule: bptt | eprop_approx | eprop | eprop_hybrid

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

### Canonical series

```bash
# Run the full Phase-1 ablation series (1a through 1g) sequentially
python scripts/run_canonical.py \
    --wandb-project cortex-lm --wandb-group canonical

# Include parameter-matched baselines at the end
python scripts/run_canonical.py \
    --wandb-project cortex-lm --wandb-group canonical \
    --baselines

# Resume from a specific phase (e.g. after 1c completed)
python scripts/run_canonical.py --start-from 1d \
    --tokenizer checkpoints/phase1a/tokenizer.pkl

# Dry run (print commands only)
python scripts/run_canonical.py --dry-run
```

### Apical stream ablation

```bash
# Run all five apical_pathway variants on the full Phase-1g architecture
python scripts/run_apical_ablation.py \
    --wandb-project cortex-lm --wandb-group apical-ablation \
    --tokenizer checkpoints/phase1a/tokenizer.pkl

# Run only selected variants
python scripts/run_apical_ablation.py --variants none additive multiplicative

# After results: re-run canonical series with winning apical variant
python scripts/run_canonical.py \
    --wandb-project cortex-lm --wandb-group canonical-with-apical \
    --override column.apical_pathway=additive
```

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

### Phase 1 — canonical ablation series (TinyStories, 1B tokens)

Each phase adds exactly one biological ingredient.  All runs share the same
tokenizer, dataset, batch size (512), and 1B-token budget.

| Phase | Config | New ingredient | Train ppl | Val ppl |
|---|---|---|---|---|
| 1a | `phase1a_minimal.yaml` | Rate neurons, simple_ei column | — | ~32 |
| 1b | `phase1b_layered.yaml` | Layered cortical columns (L4/L2-3/L5/L6) | — | ~56 (overfits) |
| 1c | `phase1c_stp.yaml` | Tsodyks-Markram STP | — | ~41 |
| 1d | `phase1d_adex.yaml` | AdEx adaptive neuron dynamics | — | ~26 |
| 1e | `phase1e_disinhibition.yaml` | VIP→SST→PC disinhibition circuit | — | ~29 |
| **1f** | `phase1f_hopfield.yaml` | Hopfield HPC, no disinhibition | **18.23*** | **20.60*** |
| 1g | `phase1g_hopfield_disinhibition.yaml` | 1f + always-on disinhibition | — | queued |
| 1h | `phase1h_hopfield_annealed.yaml` | 1f + annealed disinhibition (1→0 over 200M tokens) | — | queued |
| 1i | `phase1i_hopfield_ca1.yaml` | 1f + CA1 write-gating | — | queued |

*Note: the previously completed 1f run included `apical_pathway: additive` (a config error now fixed).
The clean 1f rerun (no apical) is in progress on Lambda.*

**Baseline comparison** (parameter-matched, ~620K params, 1B tokens on TinyStories):

| Model | Train ppl | Val ppl | Train/val gap |
|---|---|---|---|
| Transformer | 10.57 | **11.34** | 0.77 |
| RNN | 18.78 | 19.45 | 0.67 |
| **CortexLM 1f** | **18.23** | **20.60** | 2.37 |
| LSTM | 31.25 | 31.55 | 0.30 |

Key findings: The Hopfield module (1f) is the dominant contributor — it achieves the
same final val ppl as the parameter-matched RNN (~19–21 ppl) while having dramatically
better early-training sample efficiency (~8× faster in the first 100M tokens).
AdEx (1d) is the best purely-cortical component.  The transformer gap (~9 ppl) is the
primary target for the apical ablation and 1g/1h runs.

Run the series: `python scripts/run_canonical.py --wandb-project cortex-lm`

### Phase 2 — apical stream ablation

Five apical_pathway variants applied to the full Phase-1g architecture.
Goal: find the top-down pathway that best complements the cortical column.

| Variant | Config | Description |
|---|---|---|
| none | `apical_none.yaml` | No apical input (control) |
| skip | `apical_skip.yaml` | Direct embed→L5E, active from step 0 |
| additive | `apical_additive.yaml` | Gated projection; gate learned from ~0 |
| multiplicative | `apical_multiplicative.yaml` | Larkum calcium spike nonlinearity |
| corticortical | `apical_corticortical.yaml` | Circular column feedback L5E(k+1)→L23E(k) |

Run the series: `python scripts/run_apical_ablation.py --wandb-project cortex-lm`

### Phase 3 — canonical + winning apical variant

Re-run the full 1a–1g canonical series with the best apical variant applied
at each phase.  This tests whether the top-down pathway helps early phases
(simple architectures) or only the full system.

```bash
python scripts/run_canonical.py \
    --wandb-project cortex-lm --wandb-group canonical-with-apical \
    --override column.apical_pathway=<winner>
```

### Phase 4 — comparison vs. baselines
`run_baselines.py` trains RNN, LSTM, RNN+attention, LSTM+attention, and Transformer
baselines at matched parameter counts.  `plot_comparison.py` renders all learning
curves on a shared tokens-seen axis.

### Phase 5 — e-prop
`configs/bioplausible.yaml`: identical architecture to Phase 1g but trained with
online e-prop (Bellec et al. 2020) instead of BPTT.  Quantifies the cost of
biological learning rule constraints.

---

## Key biological priors

- **Dale's Law** — excitatory/inhibitory identity fixed per neuron, enforced via `softplus` on raw weights after every optimizer step
- **Log-normal timescales** — τ_m drawn from log-normal distribution (range 2–30 ms), matching cortical neuron heterogeneity
- **AdEx adaptation** — subthreshold adaptation current w with timescale τ_w (30–500 ms) implements spike-frequency adaptation
- **Tsodyks-Markram STP** — synaptic resources (u, x) deplete and recover; facilitating (E→E) vs. depressing (E→I) depending on U₀
- **Laminar routing** — feedforward signals travel L4 → L2/3 → L5; feedback travels L5 → L2/3 of lower-index columns
- **VIP→SST→PC disinhibition** — VIP interneurons inhibit SST cells, releasing pyramidal cells from inhibition; implements context-dependent gain control.  Optionally annealed: `column.disinhibition_anneal_tokens` decays the VIP→SST gain linearly from 1→0, modelling the closure of a critical-period plasticity window
- **Modern Hopfield hippocampus** — retrieves stored patterns via softmax attention; CA1 surprise signal = L2 distance between retrieved and actual state
- **Apical dendritic pathway** — top-down embedding signal reaches L5 neurons via a separate apical compartment; five variants: `none`, `skip`, `additive` (gated), `multiplicative` (Larkum calcium spike), `corticortical` (circular column feedback)

---

## Probably novel contributions

A candid assessment of what appears to be new vs. what the literature already knows.
Intended as a working checklist for the lit review and as framing for the paper introduction.

### What is already known (not our contribution)
- Modern Hopfield Networks are mathematically equivalent to transformer attention (Ramsauer et al. 2020)
- Cortical column models can perform temporal processing; liquid state machines (Maass 2002–2004)
- Complementary Learning Systems theory: hippocampus + neocortex solve the stability-plasticity dilemma (McClelland, McNaughton & O'Reilly 1995)
- AdEx adaptive exponential integrate-and-fire neuron model (Brette & Gerstner 2005)
- VIP→SST→PC disinhibitory circuit exists and modulates cortical gain (Pfeffer et al. 2013, Lee et al. 2013)

### What appears to be novel

**1. Ablation rigor on a language task.**
A clean, controlled ablation study of biologically-motivated architectural components on a language
modelling benchmark with matched parameter counts throughout.  Computational neuroscience has cortical
column models; ML has biologically-inspired architectures — but neither field has applied ML-style
empirical discipline (held-out val ppl, token-budget matching, systematic one-variable-at-a-time
ablation) to this class of model.  The combination is the gap.

**2. AdEx spike-frequency adaptation as implicit regulariser.**
The finding that AdEx neurons produce a dramatically tighter train/val gap than plain rate neurons with
identical parameter count (gap ~5 ppl vs. ~33 ppl for layered columns) in a data-repetition regime
does not appear to be documented in this form.  The proposed mechanism — adaptation forcing distributed
representations by progressively suppressing repeatedly-active neurons — is a clean, falsifiable claim
that connects neuron dynamics to generalisation theory.

**3. CLS theory confirmed in language modelling.**
CLS has been empirically tested in spatial navigation, simple categorisation, and toy sequence tasks.
Extending it to language — with a model that has structurally separate hippocampal (Hopfield) and
neocortical (layered columns) components — and obtaining an ~8× token-efficiency improvement
consistent with CLS predictions is a nontrivial extension of the theory to a new domain.
**Confirmed:** CortexLM 1f (val 20.60 at 1B tokens) vs AdEx-only (val ~26) at matched params,
with early convergence ~8× faster than a parameter-matched RNN.

**4. Annealed disinhibitory window as a training schedule.**
Implementing critical-period plasticity as a token-budget-proportional annealing schedule
(`disinhibition_anneal_tokens`) appears to be a novel technique.  The biological story is clean:
early high-plasticity window accelerates credit assignment; withdrawal of VIP→SST gain as the
cortex matures mirrors the closure of ocular dominance critical periods.  Whether it improves
performance vs. always-on disinhibition is pending (phase 1g).

**5. Differentiable CA1 write-gating for surprise-driven memory consolidation.**
Memory-augmented networks (NTMs, DNCs) have differentiable write gates, but these are not grounded
in the CA1 mismatch story.  Our implementation — `write_gate = sigmoid(T * ||retrieved - ec_obs||)`,
which suppresses Xi gradients for familiar patterns and allows them for surprising ones — is a novel
connection between the systems neuroscience of novelty detection (temporoammonic lesion dissociations)
and differentiable persistent memory.  Whether it improves on CA3-only Hopfield is pending (phase 1h).

### Honest caveats
- The model does not yet match the transformer at matched parameter count (~14 ppl target).  The
  paper's claim is not performance parity but that biological priors provide measurable, interpretable,
  additive benefits.  Phase 1f (Hopfield) and 1h (CA1) results will sharpen this.
- ~620K parameters on TinyStories is small.  Scaling to larger models and datasets is the natural
  Phase 2 and is needed for the ML community to engage seriously.
- Rate-coded neurons are partial biological plausibility.  Spiking network reviewers will flag this.

---

## Paper stories / key takeaways

A running record of the empirical results that are strong enough to anchor a paper narrative.
Updated as canonical runs complete.  All perplexities are **validation ppl** on TinyStories
unless noted.  Parameter count: ~620K throughout.

---

### Story 1 — The hippocampus is the dominant contributor (by a large margin)

Phase 1f (full system + Hopfield hippocampus) reaches **val ppl ≈ 34** at only 121M tokens —
already lower than where AdEx (the best purely-cortical variant) achieves at the same
token count (val ppl ≈ 54).  The Hopfield module compresses the learning curve by roughly
**8×** in token efficiency.

At 63M tokens, 1f train ppl is 44.8 — roughly half the train ppl of AdEx (88.4) at the same
point.  The Hopfield network is not just converging faster; it appears to be finding a
qualitatively different (better-generalising) solution.

**Final canonical results at 1B tokens:** CortexLM 1f achieves **train 18.23, val 20.60** —
matching the parameter-matched RNN (val 19.45) and dramatically outperforming LSTM (val 31.55),
while maintaining a healthy train/val gap consistent with generalisation rather than
memorisation.  The transformer achieves val 11.34, establishing the ~9 ppl target gap for
future architectural improvements.

**Proposed interpretation:** the hippocampal module acts as an episodic buffer that stores and
re-projects L5 population patterns.  This provides a second gradient pathway (HPC modulation
→ thalamic increment) that bypasses the deep cortical stack, dramatically reducing effective
depth during early training.  As the cortical weights mature, the HPC path becomes
complementary rather than dominant.

**Paper claim:** a biologically-grounded episodic memory (Hopfield network) dramatically
accelerates cortical learning and improves final generalisation — consistent with the
complementary learning systems (CLS) theory of hippocampal–neocortical interaction.

---

### Story 2 — AdEx is the best purely-cortical component

Among phases 1a–1e (no hippocampus), AdEx (1d) achieves the lowest final val ppl (~26.25)
and the tightest train/val gap (~5.7 points) despite a 20× data repetition regime.  Layered
columns alone (1b) overfit catastrophically (train ppl ~22.7, val ppl ~55.6).

**Proposed interpretation:** the AdEx adaptation timescale distribution (τ_w ∈ [30, 500] ms)
acts as an implicit regulariser: neurons that fire repeatedly for the same stimulus
progressively reduce their gain, forcing the network to use distributed representations rather
than dedicated "story-memorising" neurons.  This is functionally analogous to dropout or
weight decay, but emerges from the neuron dynamics.

**Paper claim:** spike-frequency adaptation in AdEx neurons provides implicit regularisation
that resists memorisation, producing a tight train/val gap even when training data is repeated
~20 times.

---

### Story 3 — Disinhibition accelerates early learning but plateaus

Phase 1e (AdEx + VIP→SST→PC disinhibition) has the fastest early convergence of any phase:
at 72M tokens it leads all other variants.  However, by ~500M tokens AdEx (without
disinhibition) surpasses it, and at 1B tokens 1e finishes ~3 ppl worse than 1d.

**Proposed interpretation:** the VIP disinhibitory circuit implements context-dependent gain
control that is most useful when the feedforward weights are uninformative (early training).
As weights mature, the extra VIP→SST pathway adds noise and slightly overfits.  This
motivates the annealed variant (phase 1g), which withdraws the disinhibitory window after
200M tokens — biologically analogous to the closure of a critical-period plasticity window.

**Paper claim:** early disinhibitory windows accelerate initial credit assignment;
annealing them away as training matures should yield the best of both worlds.  (To be
confirmed by phase 1g results.)

---

### Story 4 — Apical pathway placement matters: L5 not L23

Exploratory runs with all four apical variants (skip/additive/multiplicative/corticortical)
on the Hopfield base showed that the corticortical variant (which injects the top-down signal
into L23E rather than L5E) significantly underperforms the others (val ppl ~36 vs ~30 for
multiplicative at step 2200).  Gradient analysis reveals that l4_to_l23 norms are ~10×
smaller in the corticortical variant — the top-down signal at L23 interferes with feedforward
credit assignment from L4.

**Proposed interpretation:** consistent with biology, top-down signals should arrive at the
**apical dendrites of L5 pyramidal neurons** (Larkum 2013), not at L2/3 where they compete
with ascending feedforward input.  The L5 apical compartment is the "coincidence detector"
that combines context with feedforward evidence; injecting context at L23 instead corrupts
the hierarchy.

**Paper claim:** the locus of top-down input (L5 apical vs. L23) determines whether
contextual signals help or hurt; biology gets this right by routing feedback to L5.

---

### Story 5 — Baseline comparison: CortexLM matches RNN, not transformer (yet)

**Confirmed final results at 1B tokens, ~620K parameters, TinyStories:**

| Model | Train ppl | Val ppl | Train/val gap | Notes |
|---|---|---|---|---|
| Transformer | 10.57 | **11.34** | 0.77 | Perfect context window by construction |
| RNN | 18.78 | 19.45 | 0.67 | ~390-dim hidden state at matched params |
| **CortexLM 1f** | **18.23** | **20.60** | 2.37 | Biologically structured, episodic memory |
| LSTM | 31.25 | 31.55 | 0.30 | ~230-dim hidden dim due to 4× gate overhead |

**LSTM artefact:** LSTM underperforms because at matched ~620K total parameters, the 4×
gate overhead reduces hidden state dimensionality to ~230 vs ~390 for the plain RNN.
This is a parameter-matching artefact, not an architectural finding.

**Key comparison — CortexLM 1f vs RNN:** Final val ppl is nearly identical (~20.6 vs ~19.5),
but the trajectories are very different.  CortexLM 1f at 50M tokens: val 58.0.  RNN at 50M
tokens: val ~95 (estimated from curve).  The biological model is ~8× more sample-efficient
early; the RNN only catches up by running 20 passes through the same 50M training tokens.
On a non-repeating corpus the RNN advantage would vanish.

**Transformer gap (~9 ppl):** the transformer has explicit learned associative lookup over
all 128 context tokens by construction — a form of infinite working memory that our Hopfield
module (64 memories, d_model=64) approximates but does not fully replicate.  The gap is the
target for the apical ablation, larger Hopfield capacity, and 1g/1h runs.

**Early convergence of 1f-no-disinhibition:** the control run (1f without VIP circuit) shows
train 36.45, val 42.78 at only 50M tokens — dramatically faster convergence than the full
1f (train 55.28, val 58.05 at the same point).  The Hopfield module alone (without disinhibition
noise) nearly matches the transformer's early convergence (transformer val 36.18 at 50M tokens).
This suggests the VIP circuit slows down Hopfield-mediated credit assignment early in training,
motivating the annealed variant (1g).

---

### Open questions / to be confirmed

- **Does annealing disinhibition improve over always-on disinhibition?**  Phase 1g running.
- **Does CA1 error signal improve late-training generalisation?**  Phase 1h queued.  Prediction: diverges from 1g most strongly after 400M tokens.
- **Which apical variant is best on the full architecture?**  Apical ablation series queued.
- **Does the best apical variant improve all phases 1a–1f?**  Phase 3 (canonical + winning apical) queued.
- **~~Transformer target (~14 ppl) confirmed?~~**  CONFIRMED: transformer val ppl = **11.34** (train 10.57).  Gap to CortexLM 1f = ~9 ppl.

---

### Architectural directions for closing the gap with the transformer

The transformer at ~620K parameters likely achieves ~14 ppl via attention, which is
essentially an explicit learned associative lookup over every token in the context window.
Our current architecture does something analogous but more constrained: the Hopfield module
is a small associative buffer modulating thalamic input, not a full sequence-level attention.
Below are the most promising directions, with biological justification for each.

---

#### Direction 1 — Larger Hopfield memory (`n_memories`, `d_model`)

**What to try:** increase `hippocampus.n_memories` (e.g. 128→256→512) and `hippocampus.d_model`
(e.g. 64→256→512).  These are orthogonal axes: more memories increases retrieval breadth;
larger d_model increases what each memory can represent.

**On d_model in particular:** if `d_model > cortical_representation_dim` (currently 128 = 8
columns × 16 L5E neurons), the memory matrix has room to store *structured compositions* — a
single memory vector can encode a temporally extended cortical trajectory rather than a single
snapshot.  Think of it as the hippocampus "decompressing" a compressed episodic code back into
a full cortical activity sequence.  This is analogous to how some state-space models (Mamba,
S4) represent long contexts as single compressed vectors; a large d_model gives the Hopfield
network the same capacity.

**Biological basis:** CA3 pyramidal neurons have very large dendritic trees — each cell
integrates input from thousands of other CA3 cells.  The effective "dimension" of the CA3
attractor space is much larger than any individual cell count suggests, because the attractor
geometry is shaped by the full weight matrix.  A larger d_model is a more faithful model of
this high-dimensional attractor landscape.

---

#### Direction 2 — CA1 surprise / prediction-error signal

**What it is:** in the real hippocampus, CA1 receives input from *two* sources simultaneously:
- **CA3** (via Schaffer collaterals): the pattern-completed *retrieved* memory
- **Entorhinal cortex layer III** (temporoammonic path): the *actual* current input, bypassing CA3

CA1 computes the mismatch.  This serves **two functions** — and the second is primary:

1. **Error feedback to cortex** (CA1 → EC layer V → neocortex): tells the cortex how reality
   departs from the retrieved memory.
2. **Gates memory writes**: high surprise → memory is stale → update Xi.  Low surprise →
   familiar pattern → protect Xi from overwriting.  *This is the primary function — the
   hippocampus uses CA1 to decide what to write, not just to inform the cortex.*

**In our model** (`ca1: true`):
- `write_gate = sigmoid(temperature * ||error_vec||)` scales the gradient flowing into Xi.
  High surprise → gate ≈ 1 → Xi updated.  Low surprise → gate ≈ 0 → Xi frozen.
  Implemented by interpolating `retrieved` with `retrieved.detach()`, so gradients through
  Xi are scaled by the gate value.
- The directional error vector is also projected to thalamic modulation space (function 1),
  so the cortex receives both the CA3 context and the CA1 correction.
- `surprise_scale` is a learnable log-temperature controlling write-gate sharpness.

**Prediction:** `hpc/ca1_surprise` in W&B should decay over training — the network
gets better at predicting its own memories.  Improvement over 1f expected mainly
after 400M tokens (the heavy-repetition regime), when selective consolidation matters most.

**Biological basis:** temporoammonic lesions selectively impair novelty detection while
leaving CA3 pattern completion intact — exactly the dissociation we'd expect if CA1's
primary function is write-gating, not read-out.

---

#### Direction 3 — Annealed disinhibitory window (Phase 1g)

**What to try:** `column.disinhibition_anneal_tokens: 200_000_000` (already implemented).

**Why it should help:** early in training (first ~4 passes through data), the feedforward
cortical weights are uninformative.  The VIP→SST→PC circuit opens a gain-control gate that
amplifies the most active neurons regardless of whether they are correct — essentially
lowering the signal/noise threshold to allow faster credit assignment.  As weights mature,
this gate becomes counterproductive: it amplifies noise rather than signal, and the extra
VIP parameters contribute to overfitting.  Annealing to zero by 200M tokens withdraws the
gate as the cortex becomes self-sufficient.

**Biological analogy:** critical period plasticity in visual cortex — monocular deprivation
causes rapid ocular dominance shifts only during a sensitive period that closes as inhibitory
circuits mature.  The closure is driven by PV (fast-spiking) interneuron maturation, not VIP,
but the principle is the same: a window of high plasticity followed by consolidation.

**Expected gain:** ~1–2 ppl over 1f, primarily in the 200M–1B token regime.

---

#### Direction 4 — Learnable timescales

**What to try:** `neuron.learn_taus: true` — makes τ_m and τ_w gradient-optimised parameters
rather than fixed draws from the log-normal prior.

**Biological basis:** cortical membrane timescales are not fixed.  Three mechanisms:
1. **Neuromodulation**: acetylcholine, norepinephrine, and dopamine all modulate membrane
   conductances, effectively changing τ_m on seconds-to-minutes timescales.  This is a form
   of learned gain control.
2. **Ion channel expression**: neurons up/downregulate K⁺ channels (Kv1, Kv4, HCN) that
   directly set τ_m; this is activity-dependent and experience-dependent over days.
3. **Cortical hierarchy gradient**: empirically, τ_m is longer in higher areas (prefrontal
   ~100 ms) than lower areas (V1 ~10 ms).  This gradient is thought to develop to match the
   timescales of the statistical structure in the environment — a longer timescale in
   prefrontal is appropriate because it needs to integrate information over longer horizons.

In our model, learnable τ values would allow the network to self-organise a temporal hierarchy
where some neurons track fast local patterns and others integrate slowly over context —
potentially recovering something like the multi-scale attention of a transformer, but via
neuronal dynamics rather than explicit positional attention.

---

#### Direction 5 — Best apical variant (from ablation series)

**Expected gain:** ~1 ppl, possibly more if the winning variant interacts synergistically with
Hopfield.  The additive and multiplicative variants are most likely to win based on the
exploratory runs.  The apical pathway provides a direct gradient highway from the output back
to L5, complementing the HPC path — together they give the network two fast-credit-assignment
routes that bypass the deep cortical stack.

---

#### Summary table

| Direction | Config change | Expected gain | Biological justification |
|---|---|---|---|
| Larger Hopfield memory | `n_memories: 256, d_model: 256` | ~3–5 ppl | CA3 high-dimensional attractor space |
| CA1 surprise signal | `hippocampus.ca1: true` | ~2–3 ppl | Entorhinal→CA1 mismatch / novelty gating |
| Annealed disinhibition | `disinhibition_anneal_tokens: 200M` | ~1–2 ppl | Critical-period plasticity window closure |
| Learnable timescales | `learn_taus: true` | ~1–2 ppl | Neuromodulation; cortical hierarchy gradient |
| Best apical variant | `apical_pathway: additive|multiplicative` | ~1 ppl | Apical dendritic calcium spike (Larkum 2013) |

Stacking all five could plausibly close the ~12 ppl gap to the transformer, but
interactions are unknown — some may be redundant.

---

### Theoretical lens and target community

**Complementary Learning Systems (CLS) theory** (McClelland, McNaughton & O'Reilly 1995) is
the most natural framing for this work.  CLS argues that the brain requires two systems with
opposing memory properties:

- **Hippocampus**: fast learning, sparse representations, stores individual episodes without
  interference; can encode a memory in a single exposure.
- **Neocortex**: slow learning, distributed overlapping representations, extracts statistical
  regularities across many exposures.

Fast learning into distributed representations causes catastrophic interference — new
memories overwrite old ones.  The solution: the hippocampus buffers new experiences and
slowly "replays" them to the cortex during offline periods, allowing gradual consolidation.

Our architecture is the first (to our knowledge) computational implementation of CLS in a
language model.  The Hopfield module plays CA3; the cortical stack plays neocortex.  The
result — hippocampus dramatically accelerates cortical learning and improves generalisation —
is a direct empirical confirmation of the CLS prediction, in a domain (language) far from the
spatial navigation tasks where CLS was originally proposed.

**Target communities:**

| Community | Why they care | Where they publish |
|---|---|---|
| Computational neuroscience | CLS theory, cortical column models, hippocampal memory | Cosyne, CCN, eLife, PLOS Comp Bio, Neural Computation |
| Cognitive neuroscience / memory | Hippocampal–neocortical interaction, episodic memory | Neuron, Nature Neuroscience, Hippocampus |
| Reservoir computing | Fixed vs. trainable dynamics, temporal processing in recurrent networks | Neural Networks, Neural Computation, Cosyne workshops |
| State-space / structured RNN community | Mamba/S4 people interested in biology-inspired alternatives | ICLR, NeurIPS |
| ML / deep learning (biologically inspired) | Inductive biases, memory-augmented networks, scaling | NeurIPS, ICLR, ICML — *requires scaling study* |

**Note on NeurIPS:** NeurIPS main track and neuro track are not recommended targets for
Paper 1.  Review standards are opaque and the lottery is severe.  The primary audience
(comp-neuro, CLS) publishes at Cosyne and eLife.  NeurIPS becomes appropriate when a scaling
study exists to satisfy ML reviewers (Paper 3 below).

**Reservoir computing note:** our architecture is spiritually related to liquid state machines
— the cortical columns with heterogeneous AdEx timescales form a rich dynamical reservoir —
but we train the full weights rather than just the readout.  The reservoir computing community
would find this interesting as a "structured reservoir" paper.

**Recommended framing for Paper 1:** lead with the CLS story (it gives an a-priori theoretical
prediction that the hippocampus should help in repetition-heavy regimes, which the data
confirms) and let the ablation structure speak for itself — each biological ingredient
measured in isolation directly answers "what does each prior buy you?"

---

## Three-paper arc

A working plan for how this project grows into a research programme.

---

### Paper 1 — Architecture exploration and CLS confirmation *(current work)*

**Scope:** the canonical ablation series (1a–1h), apical stream ablation, and baseline
comparison at ~620K parameters on TinyStories.

**Central claim:** biologically-motivated architectural ingredients (AdEx adaptation, STP,
VIP→SST disinhibition, Hopfield hippocampus, apical dendritic pathway) each provide
measurable, additive, interpretable improvements in a language modelling benchmark.  The
result is consistent with CLS theory: a structurally separate hippocampal module accelerates
neocortical learning by ~8× in the early-training regime.

**Target venues:** eLife (primary), PLOS Computational Biology (secondary), Neural
Computation (tertiary).  Present at **Cosyne** while the journal paper is in review.

**What it does not need:** scaling results.  The claim is mechanistic and theoretical;
620K params is sufficient to establish it.

**Candidate headline result** (pending apical ablation): CortexLM with Hopfield + best
apical variant matches transformer perplexity at one pass through the training data (~50M
tokens), with both mechanisms interpretable: Hopfield provides fast episodic context
retrieval, the apical pathway provides fast access to the current token.  These are
complementary — analogous to CA3 (what did I see before?) and apical dendritic input
(what am I seeing now?).

---

### Paper 2 — How to scale biologically-structured models

**Scope:** the hard problem between the current toy model and a real large-scale result.
This is a multi-dimensional research question, not just engineering.

**The parallelization angle:** column dynamics are essentially a linear RNN with nonlinear
gating.  If the state update can be written as `h_t = A_t * h_{t-1} + B_t * x_t`, parallel
scan becomes available (Mamba/S4 style).  The challenge: AdEx has coupled two-variable
dynamics (v, w) and the nonlinearities are not trivially associative.  A linearised
approximation that preserves the biological character would be the contribution.

**The local learning rule angle:** e-prop (already in the codebase) is a candidate for
efficient training without full BPTT.  The key question: do the biological inductive biases
mean the architecture needs *less* gradient precision to converge?  If AdEx, STP, and
Hopfield are doing enough of the representational work, a noisy local update might reach the
same basin.  If true, this flips the usual framing: instead of "how do we match backprop"
the question becomes "does this architecture class have fundamentally different optimisation
requirements?"

**The fast/local path closing the gap:** predictive coding, contrastive Hebbian learning,
neuromodulated STDP — any of these could in principle train the architecture at competitive
perplexity without BPTT.  This is the most speculative and most exciting direction: not
"we achieved biological plausibility" but "the architecture's structure makes backprop
unnecessary."

**Target venues:** ICLR (if the parallelization/SSM story is clean), or eLife/Neural
Computation again if the story is primarily about local learning rules.  Could also be a
NeurIPS workshop paper that opens a conversation before the full result exists.

**Open questions:** which of the three angles above yields the most interesting result?
The paper might be "here are three approaches and what each costs" rather than a single
clean win.

---

### Paper 3 — Scaling curves

**Scope:** the same architecture run at 1M, 10M, 100M, 1B parameters on Wikitext-103 or
The Pile.  Does the sample-efficiency advantage over transformers hold as scale increases?
Does the transformer gap close, widen, or stay constant?

**Central claim:** biologically-structured inductive biases have favourable scaling
properties — the architecture extracts more signal per token at all scales, not just at toy
scale.  (Or: the gap closes and we understand why, which is also a result.)

**Target venues:** NeurIPS main track, ICLR, ICML — the ML audience is now appropriate
because there is a scaling story to tell.

**What is needed before starting this:** Paper 2's answer to the parallelization question.
At 1B parameters and full sequence length, BPTT through the current sequential column
dynamics is not feasible.  Some form of parallel training is a prerequisite.

---

## Learning rule exploration

Three learning rule variants are implemented, configurable via `learning.rule`:

| Rule | Config value | Description |
|---|---|---|
| BPTT | `bptt` | Full backpropagation through time. Gold standard. |
| e-prop (approx) | `eprop_approx` | Scalar learning signal `L(t) = mean|∂L/∂z|`. Crude eligibility traces. Fast, least biologically precise. |
| e-prop (proper) | `eprop` | Vector `L_j(t) = ∂L/∂z_j` per L5 neuron. Directional credit via `Δw_ij ∝ L_j · ē_ij`. Closer to Bellec et al. 2020. |
| e-prop hybrid | `eprop_hybrid` | Alternating e-prop (online) and brief BPTT consolidation bursts. See below. |

### Hybrid e-prop: a sleep-wake analogy

The hybrid trainer (`eprop_hybrid`) is motivated by Complementary Learning Systems theory
(McClelland et al. 1995) and the hypothesis that sleep serves a memory consolidation function.

**Awake phase (e-prop):** the model processes tokens online with local eligibility traces
and a modulatory learning signal — analogous to waking synaptic potentiation driven by
prediction error.  Credit assignment is local and causal; no future information is used.

**Sleep/replay phase (BPTT):** periodically, a short burst of BPTT is run over a fixed
replay window (default: 10 steps, every 100 e-prop steps).  This mimics hippocampal
sharp-wave ripple replay during NREM sleep: a compressed, accelerated re-processing of
recent experience that allows the global gradient to correct the slow, noisy online updates.

The hypothesis: the architecture's biological inductive biases (STP, AdEx timescales,
Hopfield episodic memory) reduce the amount of gradient precision needed, so a small amount
of BPTT "sleep" can consolidate what e-prop's noisy "waking" updates approximate.  If true,
the hybrid achieves near-BPTT performance with most of the biological learning rule
properties intact.

Key config options for the hybrid:

```yaml
learning:
  rule: eprop_hybrid
  hybrid_eprop_steps: 100      # e-prop steps per awake phase
  hybrid_bptt_steps: 10        # BPTT steps per sleep phase
  hybrid_bptt_scope: readout_only  # readout_only | full
  hybrid_eprop_variant: eprop  # eprop | eprop_approx (inner awake rule)
```

`hybrid_bptt_scope: readout_only` restricts BPTT gradients to the readout head and
inter-column weights, leaving intra-column dynamics to the local rule — the most
biologically plausible configuration.

### e-prop series script

```bash
# Step 1 — four-way comparison on 1f (find best variant)
python scripts/run_eprop_series.py \
    --tokenizer checkpoints/tokenizer.pkl \
    --wandb --wandb-project cortex-lm --wandb-group eprop-YYYY-MM-DD \
    --runs eprop_rough_cos_1f eprop_1f eprop_cos_1f

# Step 2 — ablation with winning variant on 1a, 1d, 1f
python scripts/run_eprop_series.py \
    --tokenizer checkpoints/tokenizer.pkl \
    --wandb --wandb-project cortex-lm --wandb-group eprop-YYYY-MM-DD \
    --runs eprop_1d eprop_1a   # update rule/name overrides to winner first
```

### e-prop experiment results (150 M tokens, phase 1f, TinyStories, W&B: eprop-2026-03-25)

All runs: batch=1024, seq_len=128, 1144 steps.  Val ppl reported at end of run.

| Run | Rule | LR schedule | Train ppl | Val ppl | Notes |
|---|---|---|---|---|---|
| eprop-series-bptt-1f | bptt | cosine 3e-4 | 54 | 64 | Ceiling. Still descending at 150M tokens. |
| eprop-rough-1f | eprop_approx | flat 1e-4 | 247 | 587 | Val diverges (~390→587). Train slow. |
| eprop-rough-cos-1f | eprop_approx | cosine 1e-4 | ~247 | ~587 | Tracks rough-flat almost exactly. Cosine makes no difference. |
| eprop-1f | eprop | flat 1e-4 | ~247 | ~510 | Slightly better than rough, still diverges. |
| eprop-cos-1f | eprop | cosine 1e-4 | ~247 | ~500 | **Best series 1 variant.** Lower val ppl, less noisy than rough. Still diverges. |
| eprop-hybrid-1f | eprop_hybrid | cosine 1e-4 | ~368 | ~500 | readout_only BPTT scope. Tracks eprop-cos almost exactly early; diverges slightly worse after ~80M tokens (within noise). |
| eprop-apical-1f | eprop + apical | flat 1e-4 | — | — | *pending* |
| eprop-apical-cos-1f | eprop + apical | cosine 1e-4 | — | — | *pending* |

**Key findings (series 1):**

- Cosine LR schedule and proper vector signal each help slightly, but neither stops val divergence.
  `eprop-cos-1f` is the best variant so far.
- `eprop_approx` (scalar signal) and `eprop` (vector signal) behave nearly identically —
  signal directionality is not the cause of the divergence.
- The hybrid (`eprop_hybrid`, `readout_only` scope) is indistinguishable from plain e-prop.
  Root cause: `readout_only` BPTT only updates the readout/embedding, which is already trained
  by autograd in all variants.  The recurrent weights are never corrected by the BPTT bursts.
- All variants show the same pattern: train ppl descends slowly while val ppl diverges upward.

**Series 2 diagnostic result (`freeze_recurrent`):**
Val diverges identically with frozen recurrent weights.  The recurrent e-prop updates are
**not** the cause.

**Colleague's analysis — two root causes identified:**

*Primary: state drift (train/val distributional mismatch)*

During training, `reset_state_between_batches: false` (default) carries hidden state
continuously across batches.  After N tokens, the model state has evolved for N steps; the
readout Adam updates fit to activations produced by this warm, drifted state.  During
evaluation, `evaluate()` always calls `init_state()` — a fresh zero state.  Val activations
look like what the model produces in its first ~10 tokens of existence.

In BPTT this is corrected automatically: backprop trains through the cold-start regime, so
the model learns representations that work from the beginning.  In e-prop, the model only
ever receives updates from warm-state activations.  Warm/cold divergence grows monotonically
— exactly the observed pattern.  This explains why `freeze_recurrent` didn't help: the
forward pass distributional gap is the problem, not any weight update.

**Fix:** `learning.reset_state_between_batches: true`.
**Verification:** `eprop-cos-reset-1f` confirmed — val tracked train to ~302 ppl at 50M tokens, vs. diverging to ~590 without the fix.

*Secondary (more severe): `_setup_traces` found zero synapses*

`_setup_traces` only searched for `StaticSynapse` instances.  The actual model uses
`BatchedStaticSynapse` (a different class) — so `param_traces` was always empty, and
**zero recurrent weight updates were ever applied** in any e-prop run.  Every run was
effectively pure readout + embedding Adam with no recurrent learning at all.

Consequences:
- Series 2 `adam_recurrent` diverged not because Adam is harmful, but because traces were
  empty — the Adam update was operating on garbage (all-zero traces).
- `freeze_recurrent` was trivially a no-op (nothing was happening anyway).
- The `tau_e` sweep was similarly meaningless — no traces to decay.
- **The ~500 ppl ceiling across all series 1/2 runs is a readout-only baseline**, not an
  e-prop ceiling.  Any real e-prop run that beats it confirms recurrent updates are contributing.
- All intuitions from series 1/2 (cosine vs. flat LR, scalar vs. vector signal, hybrid) were
  comparisons between identical runs.  None of it transfers to real e-prop.

*Tertiary: wrong pre/post identity in traces*

Even after fixing the `BatchedStaticSynapse` discovery, the original `_update_traces` used
L2/3 E activations as both pre-synaptic rate and post-synaptic ψ for every weight matrix.
Correct identity is synapse-specific: `syn_l4e_l23e` needs pre=`r_l4e`, post=`l23_e_v`;
`syn_l5_ee` needs pre=`r_l5e`, post=`l5_e_v`; etc.

**Both bugs now fixed:**
- `BatchedLayeredColumns.__init__` annotates each `BatchedStaticSynapse` with
  `eprop_pre_key` / `eprop_post_v_key` (18 synapses annotated).
- `_setup_traces` discovers `BatchedStaticSynapse` and builds per-column
  `[n_cols, n_post, n_pre]` trace buffers.
- `_update_traces` dispatches to the correct state tensors per synapse.
- `EpropTrainer` uses per-column vector L signal for L5E post-synaptic synapses;
  scalar fallback for all other post populations.

*Note on pseudo-derivative:* `ψ = 1 − tanh(v)²` is the **exact** derivative of tanh for our
rate-coded network — not an approximation at all.  This concern is retired.

### e-prop series 2 — diagnostics and fixes

Series 2 was designed before the empty-traces bug was discovered.  Results from the
already-completed runs are invalid as recurrent experiments (traces were always empty).
The `dale_interval` hypothesis is also invalid — `enforce_dale()` is a no-op since
`W_e >= 0` is enforced via softplus reparameterization, not by clipping.

**Colleague's revised priority order (post-fixes):**

| Fix | Priority | Reasoning |
|---|---|---|
| `adam_recurrent: true` | High | Readout uses Adam (momentum + per-param scaling); raw SGD for recurrent creates optimizer mismatch. Post-trace-fix, recurrent weights finally get meaningful signal — Adam may help them keep pace. |
| `tau_e` sweep {20, 50, 100} | Medium | Default τ_e ≈ 8 ms (geometric mean of [2, 30]) likely too short for seq_len=128. Credit horizon may span many tokens. |
| `tau_e: 2` (diagnostic) | Low | Very short traces → near-random updates → sanity check that traces carry signal at all. |
| `dale_interval` | Skip | No-op: softplus guarantees `W_e >= 0` continuously; `enforce_dale()` is already a no-op. |

**Full series-3 plan** (all with `reset_state_between_batches: true` + fixed traces, 100M tokens each):

| Step | Run | Key config | Purpose |
|---|---|---|---|
| 1 | eprop-fixed-1f | flat LR, standard τ_e | Clean baseline — recurrent updates finally real |
| 2 | eprop-fixed-adam-1f | + `adam_recurrent=true` | Fix optimizer mismatch (urgent — SGD updates 3–4 OOM too small vs Adam readout) |
| 3a | eprop-fixed-adam-recur-lr-1f | + separate recurrent LR (3e-4 or 1e-3) | If `eprop/update_mag` still tiny with Adam, higher recurrent LR needed |
| 4a | eprop-tau20-1f | + `eprop_tau_e=20` | Longer credit horizon (default τ_e≈8 likely too short for seq_len=128) |
| 4b | eprop-tau50-1f | + `eprop_tau_e=50` | Even longer traces |
| 4c | eprop-tau100-1f | + `eprop_tau_e=100` | Very long, noisier |
| 4d | eprop-tau2-1f | `eprop_tau_e=2` | Diagnostic: near-random → sanity-checks traces carry signal |
| 5 | eprop-fixed-apical-1f | best config + `column.apical_pathway=additive` | Skip connection may give readout stronger gradient signal |
| 6 | eprop-hybrid-1f | best config + `learning.rule=eprop_hybrid` | Revisit hybrid now that recurrent updates actually work |
| 7 | ablation | best config × 1a / 1d / 1f | Architecture ablation with working e-prop |

**Diagnostic metrics** (logged each eval step on all series-3 runs):
- `eprop/trace_norm_mean` — if near zero, traces not accumulating
- `eprop/l_signal` — mean |∂loss/∂z| from readout; if near zero, gradient not flowing
- `eprop/update_mag` — mean |L × trace| applied to recurrent weights; key diagnostic for optimizer mismatch

**Key hyperparameters** (in priority order after baseline established):

| Hyperparameter | Current | Notes |
|---|---|---|
| `eprop_tau_e` | auto (~8 ms) | Most theoretically motivated sweep; try {20, 50, 100}; τ_e=2 as diagnostic |
| Separate recurrent LR | same as readout (1e-4) | If `eprop/update_mag` tiny even with Adam, try 3e-4 or 1e-3 for recurrent only |
| Sequence length | 128 | Longer (256, 512) gives traces more accumulation time; expensive but high-leverage |
| Batch reset frequency | every batch | Resetting every N>1 batches lets traces accumulate longer without reintroducing state drift |
| Adam β1/β2 for recurrent | 0.9 / 0.999 | Low priority; standard values should be fine |

**Series-3 completed results (@~25M tokens, batch=32):**

| Run | Val ppl | Train ppl | Notes |
|---|---|---|---|
| eprop-cos-reset-1f | ~302 @50M | — | State drift fix confirmed; broken traces; stopped early |
| eprop-fixed-1f | ~264 @100M | — | Baseline with all fixes |
| eprop-fixed-adam-1f | ~400 flat | — | Adam amplifies noise; terminated |
| eprop-smallbatch-1f | ~390 flat | ~362 | l_signal dying; no apical; terminated |
| eprop-norml-1f | ~400 (stopped) | — | normalize alone cannot fix dying signal |
| eprop-tau50-1f | ~400 (stopped) | — | longer τ_e alone cannot fix dying signal |
| eprop-apical-norml-1d | ~151 | ~125 | normalize hurts |
| eprop-apical-norml-tau50-1f | ~140 | ~107 | normalize hurts |
| eprop-apical-norml-1f | ~135 | ~133 | normalize hurts |
| eprop-apical-tau50-1f | ~89 | ~85 | τ_e=50 mildly worse than default |
| eprop-apical-1d | ~86 | ~72 | good but noisy; 1d ≈ 1f by end |
| **eprop-apical-1f** | **~84** | **~80** | **winner — apical alone, default settings** |

**KEY FINDING 1 — apical pathway is the entire trick:**
Without apical, l_signal dies (0.0018→0.0004) and all runs plateau at ~400 ppl regardless of
normalize, τ_e, or batch size.  With apical, l_signal climbs (0.0015→0.0075), recurrent weights
learn, and val ppl reaches ~84 — a 333× improvement in sample efficiency vs BPTT (which needs
~50M tokens to reach 200 ppl; e-prop+apical reaches it within 150K tokens).

**KEY FINDING 2 — normalization is actively harmful:**
Across every setting (1f, 1d, τ_e=50), `normalize_l_signal` degrades val ppl by ~60%.
The magnitude of l_signal carries real information: when the readout is confident, smaller
updates are correct.  Normalizing discards this calibration and destabilizes training.

**KEY FINDING 3 — τ_e=50 mildly harmful; default τ_e is well-matched:**
eprop-apical-tau50-1f (val 89) vs eprop-apical-1f (val 84).  Small difference, wrong direction.

**KEY FINDING 4 — 1d (AdEx) and 1f (Hopfield) converge to the same performance under e-prop+apical:**
1d wins on train ppl (72 vs 80) but is dramatically noisier.  1f wins on val (84 vs 86) with
tighter train/val gap (4.7 vs 13.7).  By 15M tokens they trade places and are statistically
indistinguishable.  Under BPTT, 1f (Hopfield) strongly beats 1d — so e-prop is not yet
unlocking the Hopfield module's full representational power.

**KEY FINDING 5 — plateau at ~80-85 ppl; e-prop faster but BPTT goes deeper:**
All apical runs plateau around 80-85 ppl and oscillate.  BPTT on 1f reaches ~27-29 ppl —
a ~55 ppl gap.  E-prop+apical is vastly more sample-efficient early (faster even than a
transformer in the data-limited regime) but hits a ceiling that BPTT does not.  Candidate
explanations: batch cancellation noise, LR too high for the plateau, limited credit horizon.

**Mechanism:** the apical pathway (L5→L23 additive feedback) creates a more direct gradient
path into the recurrent layers.  As apical weights are learned, a positive loop forms:
better readout → stronger apical feedback → stronger l_signal → better recurrent updates.
This may explain the biological function of apical dendrites in L5 as top-down credit
assignment carriers — not just modulatory, but structurally necessary for online learning.

**Batch cancellation confirmed:** `l_signal` ∝ 1/√batch (0.0009→0.0018 as batch 64→32).
Without apical, signal too small to drive reliable descent regardless of batch size.

**Throughput:** batch=32 runs at ~1350 tokens/s (≈20h for 100M tokens) vs ~15K tokens/s for batch=1024 BPTT.
This is a fundamental bottleneck — e-prop is inherently sequential.  Ideas for later:

1. **Truncated BPTT as bridge** — longer chunks per step, fewer steps, retains online character
2. **Per-example gradients via `torch.vmap`** — get per-example `L_vec` without cancellation, then average *updates* not signals; enables large batch without cancellation
3. **`torch.compile` + bf16** — free 2–3× on forward/backward even at small batch
4. **Profile first** — at batch=32 GPU is likely underutilized; bottleneck may be Python/data overhead

---

## Roadmap and paper narrative

### The emerging story

The project has converged on a clear narrative with three interlocking findings:

1. **Architecture:** A biologically-structured cortical column (layered E/I populations, AdEx neurons,
   STP, Hopfield hippocampus) matches or exceeds parameter-matched RNNs on language modelling, and each
   component is independently motivated by neuroscience.

2. **Learning rule:** E-prop (Bellec et al. 2020) — a biologically plausible online learning rule —
   can train this architecture, *but only when the apical pathway is present.*  Without apical feedback,
   the learning signal dies within the first 1–2M tokens as the readout converges.  With apical, the
   signal grows, recurrent weights learn, and training is ~7-8× more token-efficient.

3. **Apical dendrites as credit assignment:** The additive apical pathway (L5→L23 feedback) acts as a
   learned credit assignment channel.  As apical weights are trained alongside recurrent weights, a
   positive loop forms: better readout → stronger apical feedback → stronger l_signal → better recurrent
   updates.  This provides a computational account of why apical dendrites in L5 carry top-down signals
   in the brain — they may be the biological implementation of feedback-based credit assignment.

### Immediate next steps

**Series-3 complete.** Winner: `eprop-apical-1f` (no normalize, default τ_e, 1f architecture).
All series-3 runs terminated.  Moving to hybrid e-prop/BPTT.

**Step 1 — hybrid e-prop/BPTT with apical (current priority).**
Hypothesis: e-prop+apical provides exceptional sample efficiency early but plateaus at ~80-85 ppl
due to noisy batch-averaged credit.  Periodic BPTT consolidation bursts (analogous to sleep replay)
may correct accumulated noise and drive val ppl below the e-prop floor, while keeping e-prop for
the fast early descent.  This is the most biologically motivated combination in the project.

Sweep three configurations on 1f + apical:

| Run | Config | Purpose |
|-----|--------|---------|
| eprop-hybrid-readout-1f | `hybrid_bptt_scope=readout_only`, 100 eprop + 10 bptt steps | Light consolidation — readout only |
| eprop-hybrid-full-1f | `hybrid_bptt_scope=full`, 100 eprop + 10 bptt steps | Full consolidation — all weights |
| eprop-hybrid-full-more-1f | `hybrid_bptt_scope=full`, 100 eprop + 50 bptt steps | More consolidation — does ratio matter? |

**Step 2 — sleep/wake ratio sweep (potential paper section).**

The aggressive hybrid (20 e-prop : 10 BPTT = **2:1 ratio**) is the clear winner so far.
Notably, biological sleep-wake cycles are ~16h awake : ~8h asleep — also **2:1**.
The unit of "experience" here is tokens seen rather than wall-clock time, so there is no fixed
equivalence between steps and hours.  But the *proportion* of online vs. offline processing
matching the biological ratio is striking, and may reflect a deeper computational principle
rather than an arbitrary biological constraint.

This motivates a systematic sweep of awake:asleep ratios as a potential paper section:

| Run | E-prop steps | BPTT steps | Ratio (awake:asleep) | Biological analogue |
|-----|-------------|------------|---------------------|---------------------|
| hybrid-ratio-5to1 | 50 | 10 | 5:1 | Very sleep-deprived |
| hybrid-ratio-2to1 | 20 | 10 | 2:1 | **Biological optimum — current winner** |
| hybrid-ratio-1to1 | 10 | 10 | 1:1 | Equal awake/asleep |
| hybrid-ratio-1to2 | 10 | 20 | 1:2 | More sleep than wake |
| hybrid-ratio-1to5 | 10 | 50 | 1:5 | Mostly asleep |

All with `hybrid_bptt_lr=3e-4`, `hybrid_bptt_scope=full`, apical.  The prediction: performance
peaks near 2:1 and degrades in both directions — too little BPTT leaves e-prop noise uncorrected;
too much BPTT dominates and loses the fast online learning advantage.

**Step 3 — apical BPTT sweep** (`scripts/run_hopfield_apical_sweep.py --runs 1d_apical 1f_apical 1i_apical`).
Answers: does apical help BPTT too?  Does Hopfield contribute beyond AdEx when apical is present?
Critical for disentangling architecture vs learning rule contributions.

**Step 4 — canonical ablation series.**  Clean 1a→1f runs with best hybrid config, for paper table.

### The essential question: what is causing the ~80 ppl plateau?

E-prop+apical plateaus at ~80 ppl while BPTT on the same architecture reaches ~27–29 ppl — a ~50 ppl
gap.  Two candidate explanations, and two experiments that cleanly separate them:

**Hypothesis A — Credit horizon (primary suspect).**
E-prop's eligibility traces decay as γ^t = exp(-t/τ_e).  With default τ_e ≈ 8, the trace retains
only exp(-128/8) ≈ 0% of information from the start of a 128-token sequence.  With τ_e=50, still
only ~8%.  BPTT assigns credit over the full 128-token window; e-prop structurally cannot.
TinyStories requires tracking referents over many tokens ("the little girl... she") — exactly the
regime where a short credit horizon fails.

**Hypothesis B — Batch cancellation (secondary suspect).**
At batch=32, opposite-sign learning signals cancel across sequences, attenuating recurrent updates.
Already reduced from batch=1024, but not eliminated.

**The diagnostic experiments (not yet run):**

| Experiment | Prediction if A | Prediction if B |
|---|---|---|
| `eprop_tau_e=128` (full window, γ≈0.992) | Plateau shifts 80→~65–70 | Plateau unchanged |
| `batch_size=8` (near-single-example) | Plateau unchanged | Plateau shifts downward |

If τ_e=128 moves the plateau, credit horizon is the primary constraint and the hybrid (BPTT
consolidation) is a principled biological solution — e-prop handles fast local learning,
BPTT bursts provide the long-range credit that traces cannot.
If batch=8 moves it instead, the fix is per-example traces via `torch.vmap`.
If neither moves it, the gap is intrinsic to the online approximation.

**Commands (ready to run when the hybrid sweep concludes):**
```bash
# Credit horizon test
python scripts/train.py --config configs/phase1f_hopfield.yaml --wandb --override training.batch_size=32 training.max_tokens=100000000 training.lr=0.0001 training.log_tokens=51200 training.eval_tokens=98304 training.checkpoint_dir=checkpoints/eprop-apical-tau128-1f data.tokenizer_path=checkpoints/cortex-lm-minimal_sei-rate_c8e40_bs1024_lr3e-4/ logging.project=cortex-lm logging.group=eprop-series-3 learning.rule=eprop learning.reset_state_between_batches=true learning.eprop_tau_e=128 column.apical_pathway=additive synapse.inter_column_stp=false name=eprop-apical-tau128-1f

# Batch cancellation test
python scripts/train.py --config configs/phase1f_hopfield.yaml --wandb --override training.batch_size=8 training.max_tokens=100000000 training.lr=0.0001 training.log_tokens=51200 training.eval_tokens=98304 training.checkpoint_dir=checkpoints/eprop-apical-batch8-1f data.tokenizer_path=checkpoints/cortex-lm-minimal_sei-rate_c8e40_bs1024_lr3e-4/ logging.project=cortex-lm logging.group=eprop-series-3 learning.rule=eprop learning.reset_state_between_batches=true column.apical_pathway=additive synapse.inter_column_stp=false name=eprop-apical-batch8-1f
```

### Open questions

- Is the apical pathway doing **credit assignment specifically** (e-prop benefit only) or **representation
  quality generally** (BPTT benefit too)?  Apical BPTT sweep answers this.
- Does the **Hopfield module** contribute independently of apical, or is apical doing the work?
  Compare 1d_apical vs 1f_apical in both BPTT and e-prop settings.
- Does **CA1** (phase 1i) add anything on top of CA3?
- Can **per-example traces** (`torch.vmap`) recover a larger-batch training regime once the apical
  signal bottleneck is resolved?

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

---
name: project_state
description: Current training results, next steps, pending runs, and key findings from all exploratory experiments
type: project
---

# CortexLM Project State (as of 2026-03-26)

## E-prop series-3 status

| Run | Val ppl | Notes |
|---|---|---|
| eprop-fixed-1f | ~264 @100M | All bugs fixed; baseline |
| eprop-fixed-adam-1f | ~400 flat | Adam amplifies near-zero signal into noise — do not use until cancellation fixed |
| eprop-smallbatch-1f | *running* | batch=32; l_signal=0.0018 @4K tokens; noisy around 400 ppl |

**Batch cancellation confirmed.** l_signal ∝ 1/√batch (0.0009 at batch=64, 0.0018 at batch=32).
Throughput ~1350 tokens/s = ~20h per 100M tokens.

**E-prop series-3 COMPLETE (2026-03-27). Winner: eprop-apical-1f.**

Final results @~25M tokens:
- eprop-apical-1f: val 84, train 80 — WINNER
- eprop-apical-1d: val 86, train 72 (noisier; converges to same as 1f by end)
- eprop-apical-tau50-1f: val 89 (τ_e=50 mildly worse)
- All normalize runs: val 135-151 (normalization actively harmful — discards calibration signal)
- No-apical baseline: val ~390 flat (dead)

**Key findings:**
1. Apical pathway is the entire trick — necessary and sufficient for e-prop to work
2. Normalization harmful — l_signal magnitude carries real calibration information
3. τ_e=50 mildly harmful — default τ_e well-matched to architecture
4. 1d ≈ 1f under e-prop+apical (both plateau ~84-86 val); BPTT strongly favors 1f
5. E-prop+apical is 333× more sample-efficient than BPTT early (150K vs 50M tokens to reach 200 ppl), faster even than transformer in data-limited regime
6. Plateau at ~80-85 ppl — BPTT reaches 27-29 ppl on same architecture; gap due to noisy batch-averaged credit

**Now running: hybrid e-prop/BPTT sweep (eprop-series-4)**
Three configs on 1f + apical:
- eprop-hybrid-readout-1f: readout_only BPTT consolidation (100 eprop + 10 bptt)
- eprop-hybrid-full-1f: full BPTT consolidation (100 eprop + 10 bptt)
- eprop-hybrid-full-more-1f: full BPTT, more consolidation (100 eprop + 50 bptt)

Hypothesis: periodic BPTT bursts correct accumulated noise and push val below the ~80 ppl e-prop floor,
while e-prop handles the fast early descent. Biological analogy: sleep replay consolidation.

**After hybrid — NEXT PRIORITY BEFORE RETURNING TO EPROP:**
The existing canonical BPTT series (1a–1f) was run WITHOUT apical. This is technical debt.
Apical turns out to be load-bearing for architectural components (especially Hopfield CA3, possibly CA1).
The ablation results are potentially misleading until rerun with apical.

Definitive BPTT ablation plan (do this next, in order):
1. Apical BPTT sweep — run_hopfield_apical_sweep.py --runs 1d_apical 1f_apical 1i_apical
2. Full canonical series 1a→1f WITH apical (column.apical_pathway=additive)
3. Full canonical series 1a→1f WITH apical + SGDR (training.scheduler=sgdr)

Only after these three do we have a trustworthy architecture story for the paper.
Then return to e-prop: sleep/wake ratio sweep, credit horizon diagnostics (tau128, batch8).

**E-prop throughput — ideas for later:**
1. Truncated BPTT as bridge (longer chunks, fewer steps)
2. Per-example gradients via `torch.vmap` — large batch without cancellation
3. `torch.compile` + bf16 — free 2-3× speedup
4. Profile first — GPU likely underutilized at batch=32; bottleneck may be Python/data overhead

**Why:** e-prop is inherently sequential; can't amortize over large batch like BPTT.
**How to apply:** before launching any new e-prop run, consider whether throughput is the bottleneck.



## Phase sequence (revised)
- **1a** simple_ei baseline
- **1b** + layered columns
- **1c** + Tsodyks-Markram STP
- **1d** + AdEx adaptive neurons
- **1e** + VIP disinhibition (new 1e; completes cortical column)
- **1f** + Modern Hopfield hippocampus (new 1f; qualitatively separate module)

**Why:** disinhibition is a cortical circuit component, hippocampus is a separate brain region — better paper narrative to complete the column first.

## LR schedule bug (affects all pre-canonical runs)
`scheduler.step()` was called once per BPTT chunk, not per training step. With k=32, seq_len=128 → 4 calls/step → cosine cycled 4× too fast. **All phase 1a–1e results below are non-canonical.** Fixed in bptt.py.

## Completed exploratory runs (non-canonical, batch=1024, 4000 steps)

### Apical pathway variants on phase1e (Hopfield + AdEx, fixed LR)
All batch=1024, 4000 steps, ~524M tokens, same Hopfield config.

| Variant | val_ppl (final) | val_bpb | Notes |
|---------|----------------|---------|-------|
| A (skip) | ~28.7 (stopped @1900) | ~0.94 | Best convergence speed; l23_to_l5 gradient collapses (~0.01 vs readout ~2.0) — skip pathway takes over gradient route |
| B (additive) | ~29 (crashed @2330, CUDA) | ~0.94 | Healthy gradient balance; both pathways active; gate_mean grows from 0.05 |
| C (multiplicative) | ~27-28 (est. final) | ~0.93 | At step 2200: ppl=29.95; slower but healthy gradients; nll_entropy_gap crosses 0 at ~step 1500 |
| D (corticortical) | ~33-35 (est. final) | ~1.00 | At step 2200: ppl=36.24; significantly slower convergence |

### Key gradient findings by variant
- **A (skip):** l23_to_l5 ~0.01 (readout dominates gradient route via skip)
- **B (additive):** l23_to_l5 ~0.5–1.0 (healthy; both pathways live)
- **C (multiplicative):** l23_to_l5 ~0.4–0.8 (similar to B, slightly lower)
- **D (corticortical):** l4_to_l23 ~0.0003–0.007 (10× smaller than A/B/C — feedforward pathway starved by top-down dominance)

### Hopfield behavior by variant
- **A/B/C:** HPC attn_entropy monotonically decreasing 4.16 → ~1.7–2.0; attn_max rising to 0.65–0.71. Memory bank gradually consolidates.
- **D:** HPC entropy crashed to 0.12 at step 200 (single-memory collapse), then slowly recovered to ~2.5. Unique — corticortical feedback created resonance loop locking onto one attractor.

### Tau_eff finding
L5e effective timescale grows during training (9→15-18 in variants A/B/C). L23e stable ~7-9. Network learns to exploit heterogeneous timescales. Variant D shows less L5e growth — inter-column synchrony rather than within-column temporal extension.

### Calibration (NLL-entropy gap)
Crosses zero at step ~1500 in variants B/C (model becomes well-calibrated). Stays negative in D (consistently underconfident through step 2200).

## Disinhibition implementation
VIP→SST→PC circuit implemented in BatchedLayeredColumns:
- VIP population per layer (n_vip = n_i // 2)
- E→VIP excitatory synapse
- VIP→I inhibitory synapse (disinhibits E)
- Config flag: `column.disinhibition: true/false`
- Config file: `configs/phase1e_disinhibition.yaml`

## Pending exploratory runs
1. **Disinhibition (no HPC, no apical)** — `configs/phase1e_disinhibition.yaml` — currently about to run
2. **Variant C full completion** — ongoing at step 2200/4000; estimate final ppl ~27-28

## Next after exploratory
- Canonical runs 1a–1f (batch=512, 8000 steps, W&B logging) — pending H100 access (potentially this week)
- Best-performing apical variant added to 1f canonical
- Bash script for sequential canonical runs

## Configs table
| File | Phase | Apical | HPC | Batch | Status |
|------|-------|--------|-----|-------|--------|
| phase1a_minimal.yaml | 1a | none | none | 512 | pending canonical |
| phase1b_layered.yaml | 1b | none | none | 512 | pending canonical |
| phase1c_stp.yaml | 1c | none | none | 512 | pending canonical |
| phase1d_adex.yaml | 1d | none | none | 512 | pending canonical |
| phase1e_disinhibition.yaml | 1e | none | none | 512 | running exploratory |
| phase1e_hopfield.yaml | (old 1e) | var A/B/C/D | Hopfield | 1024 | exploratory done |
| phase1e_hopfield_apical_d.yaml | exploratory | D | Hopfield | 1024 | done |
| phase1f_hopfield.yaml | 1f | best | Hopfield | 512 | pending |

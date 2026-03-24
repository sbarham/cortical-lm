---
name: project_state
description: Current training results, next steps, pending runs, and key findings from all exploratory experiments
type: project
---

# CortexLM Project State (as of 2026-03-23)

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

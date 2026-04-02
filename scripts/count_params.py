#!/usr/bin/env python3
"""
scripts/count_params.py — Detailed parameter breakdown by component.

Usage:
    python scripts/count_params.py --config configs/scale_5m.yaml
    python scripts/count_params.py --config configs/phase1f_hopfield.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def count_and_print(config_path: str, overrides: dict = None):
    from cortexlm.utils.config import get_config
    from cortexlm.model import CortexLM

    config = get_config(config_path, overrides)

    # Build tokenizer minimally (just to get vocab_size)
    vocab_size = config["data"].get("vocab_size", 4096)
    print(f"\nConfig: {config_path}")
    print(f"Vocab size (from config): {vocab_size:,}")

    model = CortexLM(config, vocab_size)

    # ── Parameter groups ──────────────────────────────────────────────────────

    def _params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    breakdown = {}

    # Embedding
    breakdown["Embedding"] = _params(model.embedding)

    # Thalamic relay
    if model.thalamic_relay is not None:
        breakdown["Thalamic relay (W_relay)"] = _params(model.thalamic_relay)

    # Column synapses (all BatchedStaticSynapse / BatchedNeuronPop in columns)
    breakdown["Column synapses + neurons"] = _params(model.columns)

    # Inter-column connectivity
    breakdown["Inter-column connectivity"] = _params(model.connectivity)

    # Hippocampus
    breakdown["Hippocampus"] = _params(model.hippocampus)

    # HPC input projection
    breakdown["HPC input proj"] = _params(model.hpc_input_proj)

    # Readout (subtract tied weight to avoid double-counting)
    readout_params = _params(model.readout)
    if config.get("readout", {}).get("weight_tying", False):
        # tied weight is embedding.weight; subtract vocab_size * embed_dim
        embed_dim = config["embedding"]["dim"]
        readout_params = max(0, readout_params - vocab_size * embed_dim)
    breakdown["Readout (excl. tied weight)"] = readout_params

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peripheral = breakdown.get("Embedding", 0) + breakdown.get("Readout (excl. tied weight)", 0)
    peripheral_pct = 100.0 * peripheral / total if total > 0 else 0.0

    # ── Print table ───────────────────────────────────────────────────────────
    col_w = 38
    divider = "-" * (col_w + 24)
    print(f"\n{'Component':<{col_w}}  {'Parameters':>12}  {'% total':>8}")
    print(divider)
    for name, n in breakdown.items():
        pct = 100.0 * n / total if total > 0 else 0.0
        print(f"  {name:<{col_w-2}}  {n:>12,}  {pct:>7.1f}%")
    print(divider)
    print(f"  {'Total':<{col_w-2}}  {total:>12,}  {'100.0%':>8}")

    # ── Warnings ──────────────────────────────────────────────────────────────
    print()
    if peripheral_pct > 40.0:
        print(f"  WARN: Peripheral dominance: embedding + readout = {peripheral_pct:.1f}% of total")
        print(f"        (target: < 40%).  Note: the transformer baseline carries identical embedding")
        print(f"        overhead, so cortex-lm vs. transformer comparisons are unaffected.")
        print(f"        For within-cortex-lm scaling plots, report cortical params, not total.")

    # Check fan-in on thalamic projection
    cols = model.columns
    if hasattr(cols, "thal_proj_e_w"):
        fan_in = cols.thal_proj_e_w.shape[-1]
        if fan_in > 256:
            print(f"  WARN: thal_proj fan-in = {fan_in} > 256 (saturation risk).")

    print()
    return total


def main():
    parser = argparse.ArgumentParser(description="Parameter count breakdown for CortexLM configs")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides as key=value (e.g. column.n_columns=32)")
    args = parser.parse_args()

    def _parse_val(v):
        try: return int(v)
        except ValueError: pass
        try: return float(v)
        except ValueError: pass
        if v.lower() in ("true", "false"): return v.lower() == "true"
        return v

    def _nested_set(d, dotted_key, val):
        keys = dotted_key.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val

    overrides = {}
    for kv in (args.override or []):
        k, v = kv.split("=", 1)
        _nested_set(overrides, k, _parse_val(v))

    count_and_print(args.config, overrides if overrides else None)


if __name__ == "__main__":
    main()

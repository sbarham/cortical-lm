"""Console + WandB + JSONL file logging."""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("cortexlm")


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


class Logger:
    def __init__(self, config: dict):
        self.config = config
        self.log_cfg = config.get("logging", {})
        self.use_wandb = self.log_cfg.get("wandb", False)
        self.log_interval = self.log_cfg.get("log_interval", 100)
        self._wandb = None
        self._start_time = time.time()

        # JSONL metrics file — lives next to checkpoints so it survives the run
        ckpt_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        metrics_path = os.path.join(ckpt_dir, "metrics.jsonl")
        self._metrics_file = open(metrics_path, "a", buffering=1)  # line-buffered

        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=self.log_cfg.get("project", "cortex-lm"),
                config=self.config,
            )
        except ImportError:
            logger.warning("wandb not installed; disabling WandB logging")
            self.use_wandb = False

    def log(self, metrics: Dict[str, Any], step: int):
        elapsed = time.time() - self._start_time
        parts = [f"step={step}", f"t={elapsed:.0f}s"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        logger.info(" | ".join(parts))

        # Write to JSONL
        record = {"step": step, "t": round(elapsed, 1), **metrics}
        self._metrics_file.write(json.dumps(record) + "\n")

        if self.use_wandb and self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def finish(self):
        self._metrics_file.close()
        if self.use_wandb and self._wandb is not None:
            self._wandb.finish()

"""Configuration for the residual-CEGAR DeepAnT pipeline.

Groups three concerns:
  * DeepAnT forecasting hyperparameters (vendored defaults).
  * Residual-CEGAR gate/controller settings (the new surface).
  * W&B logging (entity / project for the RWML project).

The CEGAR numeric defaults (``k``, ``tau``) are PLACEHOLDERS pending Luis's
notebook; see :mod:`cegar.residual_signals`.

W&B configuration is read from environment variables (via .env file):
  - WANDB_ENTITY: default "yoonmeehwang-carnegie-mellon-university"
  - WANDB_PROJECT: default "RWML"
  - WANDB_GROUP: optional group name
  - WANDB_NAME: optional run name
  - WANDB_MODE: "online" (default), "offline", or "disabled"
  - WANDB_ENABLED: "1" to enable (default), "0" to disable
"""
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; fall back to environment variables


@dataclass
class DeepAnTConfig:
    window: int = 45
    pred_window: int = 1
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 500
    split: float = 0.8
    early_stopping_delta: float = 0.05
    early_stopping_patience: int = 10
    random_state: int = 42
    # CNN internals (hardcoded upstream; surfaced here for completeness)
    filter1_size: int = 128
    filter2_size: int = 32
    kernel_size: int = 2
    pool_size: int = 2
    stride: int = 1


@dataclass
class CegarConfig:
    """Residual-CEGAR gate + controller settings."""
    enabled: bool = True

    # --- gate strength (lambda) ---
    lam: float = 0.05                 # fixed lambda when lam_rule == "fixed"
    lam_rule: str = "auto_tr"         # "fixed" | "auto_tr"
    lam_max: float = 1.5
    lam_ema: float = 0.9              # EMA smoothing for auto lambda
    ratio_target: float = 3.0         # tail-ratio target r for auto_tr
    ratio_beta: float = 0.9           # EMA for gate_mean / gate_p99 stats
    invalid_decay: float = 0.95

    # --- threshold (tau) ---
    tau: float = 0.5                  # PLACEHOLDER (pending Luis's notebook)
    tau_rule: str = "fixed"           # "fixed" | "auto_q_valley"
    tau_ema: float = 0.9
    tau_valley_warmup: int = 5
    tau_valley_smooth: int = 3
    tau_min: float = 0.1
    tau_max: float = 0.99
    tau_q_start: float = 0.6          # initial quantile for auto_q_valley

    # --- confidence / wrongness signal ---
    k: float = 1.0                    # PLACEHOLDER sigmoid sharpness (pending notebook)
    conf_type: str = "residual_ema"   # reserved for future signal variants
    residual_ema_beta: float = 0.9
    residual_buffer_size: int = 5000
    detach_gates: bool = True
    scale_normalize: bool = True      # normalize scale (recommended with auto_tr)


@dataclass
class WandbConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("WANDB_ENABLED", "1").lower() in ("1", "true", "yes"))
    entity: str = field(default_factory=lambda: os.getenv("WANDB_ENTITY", "yoonmeehwang-carnegie-mellon-university"))
    project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "RWML"))
    group: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_GROUP", None))
    name: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_NAME", None))
    mode: str = field(default_factory=lambda: os.getenv("WANDB_MODE", "online"))  # "online" | "offline" | "disabled"


@dataclass
class RWMLConfig:
    deepant: DeepAnTConfig = field(default_factory=DeepAnTConfig)
    cegar: CegarConfig = field(default_factory=CegarConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # I/O
    data_input: str = "data/dataset.csv"
    model_output: str = "results/model.pt"
    score_output: str = "results/algorithm-scores.csv"

    def to_dict(self) -> Dict:
        return asdict(self)


def init_wandb(cfg: RWMLConfig):
    """Initialize a W&B run from config, or return ``None`` if unavailable/disabled.

    Never raises: if ``wandb`` is not installed or logging is disabled, training
    still runs (the train loop guards every ``wandb.log`` call).
    """
    if not cfg.wandb.enabled or cfg.wandb.mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; continuing without logging.")
        return None
    return wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        config=cfg.to_dict(),
    )

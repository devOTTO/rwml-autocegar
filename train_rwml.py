"""Residual-CEGAR DeepAnT training loop.

Wires the full path described in the plan::

    DeepAnT forward  ->  residual (y - y_hat)
                     ->  (E_t wrongness, C_t confidence)        [cegar.residual_signals]
                     ->  gate = E_t * C_t  ->  scale = 1 + lam*gate   [cegar.gate]
                     ->  ScaleGrad on y_hat  ->  L1 loss  ->  backward  [cegar.scale_grad]

Controllers update once per epoch:
    * tail-ratio lambda controller from gate (gate_mean / gate_p99) EMAs.
    * valley-detection tau controller from the accumulated E_t histogram.

Set ``CegarConfig.enabled = False`` to fall back to the vanilla DeepAnT loop
(identical to ``deepant.predictor.Predictor.train``) for the baseline control.

NOTE: the numeric E_t / C_t formulas and the tau/k constants are PLACEHOLDERS
(see :mod:`cegar.residual_signals`). The *wiring* here is final.
"""
from typing import Optional

import numpy as np
import torch
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from config import CegarConfig, DeepAnTConfig
from deepant.model import DeepAnTCNN
from deepant.early_stopping import EarlyStopping
from deepant.helper import retrieve_save_path
from cegar import (
    ScaleGrad,
    gate,
    update_ema,
    tail_ratio_lambda_controller,
    valley_quantile_controller,
    ResidualStats,
    compute_wrongness_E_t,
    compute_confidence_C_t,
)

_HIST_BINS = 50


def _wandb_log(wandb_run, payload, step=None):
    if wandb_run is None:
        return
    try:
        wandb_run.log(payload, step=step)
    except Exception:
        pass


class RWMLPredictor:
    """DeepAnT predictor with an optional residual-CEGAR gradient gate."""

    def __init__(self, deepant_cfg: DeepAnTConfig, cegar_cfg: CegarConfig, in_channels: int,
                 wandb_run=None, num_workers: int = 0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dcfg = deepant_cfg
        self.ccfg = cegar_cfg
        self.in_channels = in_channels
        self.wandb_run = wandb_run
        self.num_workers = num_workers

        seed = deepant_cfg.random_state
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.model = DeepAnTCNN(
            deepant_cfg.window, deepant_cfg.pred_window, in_channels,
            deepant_cfg.filter1_size, deepant_cfg.filter2_size,
            deepant_cfg.kernel_size, deepant_cfg.pool_size, deepant_cfg.stride,
        ).to(self.device)

        # controller state
        self._lam_smooth: Optional[float] = None
        self._gate_mean_ema: Optional[float] = None
        self._gate_p99_ema: Optional[float] = None
        self._tau_q_cur: float = cegar_cfg.tau_q_start

    # ------------------------------------------------------------------ utils
    def _current_lambda(self) -> float:
        if self.ccfg.lam_rule == "auto_tr" and self._lam_smooth is not None:
            return self._lam_smooth
        return self.ccfg.lam

    def _current_tau(self, res_stats: ResidualStats) -> float:
        """Residual threshold tau feeding C_t.

        ``fixed``: the configured ``cegar.tau``.
        ``auto_q_valley``: map the valley quantile (in E_t space) to a residual
        level via the residual buffer. (PLACEHOLDER mapping pending notebook.)
        """
        if self.ccfg.tau_rule == "auto_q_valley":
            buf = res_stats._global_buffer
            if buf:
                t = torch.tensor(list(buf), dtype=torch.float32)
                return float(torch.quantile(t, self._tau_q_cur).item())
        return self.ccfg.tau

    # ----------------------------------------------------------------- train
    def train(self, train_dataset: Dataset, valid_dataset: Dataset, save_path: str):
        if not self.ccfg.enabled:
            return self._train_vanilla(train_dataset, valid_dataset, save_path)

        model_save_name = retrieve_save_path(save_path, "model.pt")
        optimizer = Adam(self.model.parameters(), lr=self.dcfg.lr)
        criterion = L1Loss()

        loader_train = DataLoader(train_dataset, batch_size=self.dcfg.batch_size,
                                  num_workers=self.num_workers, pin_memory=True)
        loader_valid = DataLoader(valid_dataset, batch_size=self.dcfg.batch_size,
                                  num_workers=self.num_workers, pin_memory=True)

        res_stats = ResidualStats(self.in_channels, ema_beta=self.ccfg.residual_ema_beta,
                                  buffer_size=self.ccfg.residual_buffer_size)
        early_stopping = EarlyStopping(self.dcfg.early_stopping_patience,
                                       self.dcfg.early_stopping_delta, self.dcfg.epochs)

        valid_loss_min = np.inf
        for epoch in early_stopping:
            self.model.train()
            train_losses = []
            hist_accum = np.zeros(_HIST_BINS, dtype=np.float64)
            lam_cur = self._current_lambda()

            for X, y in loader_train:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                output = self.model(X)                      # [B, C, H]

                # --- residual -> E_t, C_t ---
                residual = (y - output).detach()
                window_resid = res_stats.update(residual)  # [B]
                tau_cur = self._current_tau(res_stats)
                E_t = compute_wrongness_E_t(window_resid, res_stats.q95)        # [B]
                C_t_scalar = compute_confidence_C_t(res_stats.ema, tau_cur, self.ccfg.k)
                C_t = torch.full_like(E_t, C_t_scalar)                          # [B]

                # --- gate -> scale -> ScaleGrad ---
                scale, stats = gate(
                    confidence=C_t,
                    wrongness=E_t,
                    lam=lam_cur,
                    scale_normalize=self.ccfg.scale_normalize,
                    detach_gates=self.ccfg.detach_gates,
                    conf_raw=E_t,            # histogram over E_t for valley detection
                    hist_bins=_HIST_BINS,
                )
                scaled_output = ScaleGrad.apply(output, scale.view(-1, 1, 1))
                loss = criterion(scaled_output, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                # --- accumulate controller stats ---
                self._gate_mean_ema = update_ema(self._gate_mean_ema, stats["gate_mean"], self.ccfg.ratio_beta)
                self._gate_p99_ema = update_ema(self._gate_p99_ema, stats["gate_p99"], self.ccfg.ratio_beta)
                hist_accum += np.asarray(stats["_conf_hist"], dtype=np.float64)

            train_loss = float(np.sum(train_losses))

            # --- per-epoch controllers ---
            if self.ccfg.lam_rule == "auto_tr":
                self._lam_smooth, lam_info = tail_ratio_lambda_controller(
                    self._gate_mean_ema, self._gate_p99_ema,
                    ratio_target=self.ccfg.ratio_target, lam_max=self.ccfg.lam_max,
                    prev_lam_smooth=self._lam_smooth, lam_ema=self.ccfg.lam_ema,
                    invalid_decay=self.ccfg.invalid_decay,
                )
            if self.ccfg.tau_rule == "auto_q_valley" and epoch > self.ccfg.tau_valley_warmup:
                self._tau_q_cur, valley_info = valley_quantile_controller(
                    hist_accum.tolist(), prev_q=self._tau_q_cur, ema_beta=self.ccfg.tau_ema,
                    smooth=self.ccfg.tau_valley_smooth, q_min=self.ccfg.tau_min, q_max=self.ccfg.tau_max,
                )

            # --- validation ---
            self.model.eval()
            valid_losses = []
            with torch.no_grad():
                for X, y in loader_valid:
                    X, y = X.to(self.device), y.to(self.device)
                    valid_losses.append(criterion(self.model(X), y).item())
            valid_loss = float(np.sum(valid_losses))
            early_stopping.update(valid_loss)

            _wandb_log(self.wandb_run, {
                "epoch": int(epoch),
                "train/loss": train_loss,
                "valid/loss": valid_loss,
                "CEGAR/lambda": float(lam_cur),
                "CEGAR/gate_mean_ema": float(self._gate_mean_ema or 0.0),
                "CEGAR/gate_p99_ema": float(self._gate_p99_ema or 0.0),
                "CEGAR/tau_q_cur": float(self._tau_q_cur),
                "CEGAR/residual_q95": float(res_stats.q95),
                "CEGAR/residual_ema": float(res_stats.ema),
            }, step=int(epoch))

            print(f"Epoch {epoch}/{self.dcfg.epochs}  train={train_loss:.6f}  valid={valid_loss:.6f}  "
                  f"lam={lam_cur:.4f}  q95={res_stats.q95:.4f}  res_ema={res_stats.ema:.4f}")

            if valid_loss < valid_loss_min:
                torch.save(self.model.state_dict(), model_save_name)
                valid_loss_min = valid_loss

        return model_save_name

    # --------------------------------------------------------- vanilla path
    def _train_vanilla(self, train_dataset, valid_dataset, save_path):
        """Baseline DeepAnT loop (CEGAR disabled): identical math to upstream."""
        model_save_name = retrieve_save_path(save_path, "model.pt")
        optimizer = Adam(self.model.parameters(), lr=self.dcfg.lr)
        criterion = L1Loss()
        loader_train = DataLoader(train_dataset, batch_size=self.dcfg.batch_size,
                                  num_workers=self.num_workers, pin_memory=True)
        loader_valid = DataLoader(valid_dataset, batch_size=self.dcfg.batch_size,
                                  num_workers=self.num_workers, pin_memory=True)
        early_stopping = EarlyStopping(self.dcfg.early_stopping_patience,
                                       self.dcfg.early_stopping_delta, self.dcfg.epochs)
        valid_loss_min = np.inf
        for epoch in early_stopping:
            self.model.train()
            train_losses = []
            for X, y in loader_train:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = float(np.sum(train_losses))

            self.model.eval()
            valid_losses = []
            with torch.no_grad():
                for X, y in loader_valid:
                    X, y = X.to(self.device), y.to(self.device)
                    valid_losses.append(criterion(self.model(X), y).item())
            valid_loss = float(np.sum(valid_losses))
            early_stopping.update(valid_loss)
            _wandb_log(self.wandb_run, {"epoch": int(epoch), "train/loss": train_loss,
                                        "valid/loss": valid_loss}, step=int(epoch))
            print(f"[baseline] Epoch {epoch}/{self.dcfg.epochs}  train={train_loss:.6f}  valid={valid_loss:.6f}")
            if valid_loss < valid_loss_min:
                torch.save(self.model.state_dict(), model_save_name)
                valid_loss_min = valid_loss
        return model_save_name

    # ----------------------------------------------------------- inference
    def predict(self, test_dataset: Dataset) -> torch.Tensor:
        self.model.eval()
        loader = DataLoader(test_dataset, batch_size=self.dcfg.batch_size, pin_memory=True,
                            num_workers=self.num_workers)
        result = []
        for x, _ in loader:
            x = x.to(self.device)
            with torch.no_grad():
                result.append(self.model(x).detach())
        return torch.cat(result, dim=0)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

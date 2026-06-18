"""RW-2 trainer (Afonso Baldo 2025, Algorithm 3).

Identical to RW-1 but applies a Savitzky-Golay filter to the correction
tensor at the end of each epoch.  The filter enforces temporal smoothness,
helping detect *sequential* anomalies (shapelet / trend patterns) while
suppressing the noisy point-wise corrections that cause RW-1 to fail on
datasets with point anomalies like TAO.

From the thesis results (Table 6.2):
  - RW-2 is best in 7/17 datasets, strongest gains on MSL (+4×) and SMAP (+2×)
  - Trades off accuracy on point-anomaly datasets (TAO) due to smoothing
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader, TensorDataset

from deepant.model import DeepAnTCNN
from deepant.helper import retrieve_save_path
from .rw1 import RW1Trainer


class RW2Trainer(RW1Trainer):
    """RW-1 + Savitzky-Golay smoothing on the correction tensor.

    Args:
        savgol_window: window length for the Savitzky-Golay filter (must be
            odd).  Larger = more smoothing.  Default 11.
        savgol_polyorder: polynomial order.  Default 3 (cubic).
        All other args: same as ``RW1Trainer``.
    """

    def __init__(self, deepant_cfg, in_channels: int,
                 alpha: float = 1e-2,
                 correction_lr: float = 1e-2,
                 rmsprop_alpha: float = 0.9,
                 savgol_window: int = 11,
                 savgol_polyorder: int = 3,
                 wandb_run=None,
                 num_workers: int = 0):
        super().__init__(deepant_cfg, in_channels, alpha, correction_lr,
                         rmsprop_alpha, wandb_run, num_workers)
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder

    def _smooth_correction(self):
        """Apply per-channel Savitzky-Golay filter to the correction tensor."""
        from scipy.signal import savgol_filter

        corr_np = self.correction.detach().cpu().numpy()  # [T, C]
        T, C = corr_np.shape
        wl = min(self.savgol_window, T)
        if wl % 2 == 0:
            wl -= 1                    # must be odd
        if wl < self.savgol_polyorder + 2:
            return                     # series too short to filter

        for c in range(C):
            corr_np[:, c] = savgol_filter(corr_np[:, c], wl, self.savgol_polyorder)

        with torch.no_grad():
            self.correction.copy_(
                torch.from_numpy(corr_np).float().to(self.correction.device)
            )

    # override train to inject smoothing step
    def train(self, X: np.ndarray, save_path: str):
        """Train with RW-1 + Savitzky-Golay correction smoothing (epoch-wise)."""
        model_save = retrieve_save_path(save_path, "model.pt")

        X_t = torch.from_numpy(X).float().to(self.device)
        T, C = X_t.shape

        self.correction = torch.nn.Parameter(-X_t.clone())

        opt_model = Adam(self.model.parameters(), lr=self.dcfg.lr)
        opt_corr = RMSprop([self.correction], lr=self.correction_lr,
                           alpha=self.rmsprop_alpha, eps=1e-8)

        best_loss = float("inf")

        for epoch in range(self.dcfg.epochs):
            self.model.train()

            X_corr = X_t + self.correction
            xs, ys = self._make_windows(X_corr)

            loader = DataLoader(
                TensorDataset(torch.arange(len(xs))),
                batch_size=self.dcfg.batch_size,
                shuffle=True,
            )

            opt_model.zero_grad()
            opt_corr.zero_grad()

            epoch_loss = 0.0
            n_batches = 0
            for (idx,) in loader:
                x_b = xs[idx]
                y_b = ys[idx]

                y_hat = self.model(x_b)
                pred_loss = torch.sqrt(
                    F.mse_loss(y_hat, y_b, reduction="none").mean(dim=[1, 2])
                ).mean()
                reg_loss = self.alpha * self.correction.abs().mean()
                loss = pred_loss + reg_loss
                loss.backward(retain_graph=True)
                epoch_loss += loss.item()
                n_batches += 1

            opt_model.step()
            opt_corr.step()

            # ── Savitzky-Golay smoothing ──────────────────────────────────
            self._smooth_correction()

            avg_loss = epoch_loss / max(n_batches, 1)
            corr_mag = self.correction.detach().abs().mean().item()

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), model_save)

            if epoch % max(1, self.dcfg.epochs // 20) == 0:
                print(f"[RW-2] epoch {epoch:>4}/{self.dcfg.epochs}  "
                      f"loss={avg_loss:.5f}  |correction|={corr_mag:.5f}")

            self._step_wandb({
                "RW2/loss": avg_loss,
                "RW2/correction_mag": corr_mag,
            }, epoch)

        print(f"[RW-2] done. Model → {model_save}")

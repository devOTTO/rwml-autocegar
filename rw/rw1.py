"""RW-1 trainer (Afonso Baldo 2025, Algorithm 2).

Key idea
--------
A learnable ``correction`` tensor (same shape as the raw dataset) is
initialised to ``-X`` so that ``X + correction = 0`` at the start.
During training the model learns to predict ``X + correction``, and the
correction is simultaneously updated to minimise a combined loss::

    loss = RMSE(y_hat, y_corrected) + alpha * ||correction||_1

The model is pushed to predict the corrected signal, while the L1 term
encourages the correction to be sparse.  Points the model *cannot*
predict well despite a sparse correction are implicitly anomalous.

Anomaly score
-------------
``|correction|`` averaged over the feature dimension — large correction
magnitude at a timestep signals that the algorithm decided the original
data there was anomalous.

Training mode
-------------
Epoch-wise (default): correction updated once per epoch after
accumulating gradients from all batches.  Shown to outperform
instance-wise on most datasets in the thesis.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader, TensorDataset

from deepant.model import DeepAnTCNN
from deepant.helper import retrieve_save_path


class RW1Trainer:
    """Train DeepAnT with RW-1 correction mechanism.

    Args:
        deepant_cfg: ``DeepAnTConfig`` from ``config.py``.
        in_channels: number of input features (C).
        alpha: L1 regularisation weight on correction.  Controls sparsity
            of corrections; lower = more corrections allowed.
        correction_lr: learning rate for the RMSprop correction update.
        rmsprop_alpha: RMSprop smoothing constant β.
        wandb_run: optional W&B run for logging.
    """

    def __init__(self, deepant_cfg, in_channels: int,
                 alpha: float = 1e-2,
                 correction_lr: float = 1e-2,
                 rmsprop_alpha: float = 0.9,
                 wandb_run=None,
                 num_workers: int = 0):
        self.dcfg = deepant_cfg
        self.in_channels = in_channels
        self.alpha = alpha
        self.correction_lr = correction_lr
        self.rmsprop_alpha = rmsprop_alpha
        self.wandb_run = wandb_run
        self.num_workers = num_workers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = deepant_cfg.random_state
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = DeepAnTCNN(
            deepant_cfg.window, deepant_cfg.pred_window, in_channels,
            deepant_cfg.filter1_size, deepant_cfg.filter2_size,
            deepant_cfg.kernel_size, deepant_cfg.pool_size, deepant_cfg.stride,
        ).to(self.device)

        # correction is set up in train() once we know the data shape
        self.correction: torch.Tensor | None = None

    # ---------------------------------------------------------------- helpers
    def _make_windows(self, data_2d: torch.Tensor):
        """Build (x_windows, y_windows) from full [T, C] corrected dataset."""
        T = data_2d.shape[0]
        W, P = self.dcfg.window, self.dcfg.pred_window
        xs, ys = [], []
        for i in range(T - W - P + 1):
            xs.append(data_2d[i:i + W].T)              # [C, W]
            ys.append(data_2d[i + W:i + W + P].T)      # [C, P]
        return torch.stack(xs), torch.stack(ys)         # [N, C, W], [N, C, P]

    def _step_wandb(self, payload: dict, epoch: int):
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log({**payload, "epoch": epoch}, step=epoch)
        except Exception:
            pass

    # ----------------------------------------------------------------- train
    def train(self, X: np.ndarray, save_path: str):
        """Train on the full dataset (train + test), epoch-wise.

        Args:
            X: raw data array, shape ``[T, C]``.
            save_path: path to save the best model checkpoint.
        """
        model_save = retrieve_save_path(save_path, "model.pt")

        X_t = torch.from_numpy(X).float().to(self.device)       # [T, C]
        T, C = X_t.shape

        # correction initialised to -X (so X + correction = 0 at start)
        self.correction = torch.nn.Parameter(-X_t.clone())

        opt_model = Adam(self.model.parameters(), lr=self.dcfg.lr)
        opt_corr = RMSprop([self.correction], lr=self.correction_lr,
                           alpha=self.rmsprop_alpha, eps=1e-8)

        best_loss = float("inf")

        for epoch in range(self.dcfg.epochs):
            self.model.train()

            # build windows from current corrected data (detach so graph is fresh)
            X_corr = (X_t + self.correction)               # [T, C] with grad
            xs, ys = self._make_windows(X_corr)             # need grad

            # We need gradients wrt correction: rebuild without detach
            # Use a mini-dataset from the current corrected tensor
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
                x_b = xs[idx]     # [B, C, W]
                y_b = ys[idx]     # [B, C, P]

                y_hat = self.model(x_b)                     # [B, C, P]

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

            avg_loss = epoch_loss / max(n_batches, 1)
            corr_mag = self.correction.detach().abs().mean().item()

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), model_save)

            if epoch % max(1, self.dcfg.epochs // 20) == 0:
                print(f"[RW-1] epoch {epoch:>4}/{self.dcfg.epochs}  "
                      f"loss={avg_loss:.5f}  |correction|={corr_mag:.5f}")

            self._step_wandb({
                "RW1/loss": avg_loss,
                "RW1/correction_mag": corr_mag,
            }, epoch)

        print(f"[RW-1] done. Model → {model_save}")

    # ------------------------------------------------------- anomaly score
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return per-timestep anomaly score = mean |correction| over channels.

        Args:
            X: original data ``[T, C]`` (same array passed to ``train``).

        Returns:
            ``[T]`` float array.  Large values = anomalous.
        """
        if self.correction is None:
            raise RuntimeError("Call train() before anomaly_score().")
        score = self.correction.detach().abs().mean(dim=1).cpu().numpy()
        return score.astype(np.float32)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

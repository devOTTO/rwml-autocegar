"""Anomaly score from forecast residuals (vendored, unchanged from upstream DeepAnT).

Per-window RMSE of the prediction error, then global z-score across the test
series. This is the score the AUC-ROC baseline (0.679) is computed on.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class Detector():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self, predictedY: torch.Tensor, test_dataset: Dataset, batch_size: int = 4096) -> np.ndarray:
        loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
        offset = 0
        scores_list = []
        for _, test_y_batch in loader:
            bs = test_y_batch.shape[0]
            pred_batch = predictedY[offset:offset + bs].to(self.device)
            test_y_batch = test_y_batch.to(self.device)
            s = torch.sqrt(F.mse_loss(pred_batch.detach(), test_y_batch.detach(), reduction="none").sum(dim=[1, 2]))
            scores_list.append(s.cpu())
            offset += bs

        anomaly_score = torch.cat(scores_list)
        anomaly_score = (anomaly_score - anomaly_score.mean()).abs() / anomaly_score.std()
        return anomaly_score.numpy()

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

    def detect(self, predictedY: torch.Tensor, test_dataset: Dataset) -> np.ndarray:
        predictedY = predictedY.to(self.device)

        _, test_y = next(iter(DataLoader(test_dataset, batch_size=predictedY.shape[0], pin_memory=True)))
        test_y = test_y.to(self.device)

        # calculate euclidian distance
        anomaly_score = torch.sqrt(F.mse_loss(predictedY.detach(), test_y.detach(), reduction="none").sum(dim=[1, 2]))
        # standardize error
        anomaly_score = (anomaly_score - anomaly_score.mean()).abs() / anomaly_score.std()
        return anomaly_score.cpu().numpy()

"""Sliding-window time-series dataset (vendored, unchanged from upstream DeepAnT)."""
import torch

from torch.utils.data import Dataset
from typing import List, Optional, Tuple


class TimeSeries(Dataset):
    def __init__(self, X, window_length: int, prediction_length: int, output_dims: Optional[List[int]] = None, overlap=1):
        self.output_dims = output_dims or list(range(X.shape[1]))
        self.X = torch.from_numpy(X).float().permute(1, 0)  # [C, W]
        self.window_length = window_length
        self.prediction_length = prediction_length
        self.overlap = overlap

    def __len__(self):
        if self.overlap == 1.0:
            step_size = 1
        else:
            step_size = int(self.window_length * (1 - self.overlap))

        # Calculate how many complete windows we can fit
        valid_length = self.X.shape[-1] - self.window_length - self.prediction_length + 1

        # Return ceiling division to get the number of windows
        return (valid_length + step_size - 1) // step_size

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.overlap == 1.0:
            step_size = 1
        else:
            step_size = int(self.window_length * (1 - self.overlap))

        index = index * step_size

        end_idx = index + self.window_length
        x = self.X[:, index:end_idx]
        y = self.X[:, end_idx:end_idx + self.prediction_length]
        return x, y

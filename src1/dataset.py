import torch
import numpy as np


class SimpleDataset_envpipe(torch.utils.data.Dataset):
    def __init__(self, seq, d_model, size=8):
        self._size = size
        self._inputs = np.random.randn(size, seq, d_model)
        self._labels = np.random.randn(size, seq)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        x, y = (
            torch.tensor(self._inputs[idx % self._size], dtype=torch.float32),
            self._labels[idx % self._size].astype("float32"),
        )
        return (x, y)


class SimpleDataset(torch.utils.data.Dataset):
    """
    A dataset that generates random embedded sequence and labels
    for decoder layers (no embedding layer or final classifier layer).
    """

    def __init__(self, seq, d_model, size=192):
        """_summary_

        Args:
            seq (int): Length of the sequence (context).
            d_model (int): Dimension of embedding.
            size (int, optional): Size of the dataset. Defaults to 100.
        """
        self._size = size
        self._inputs = torch.tensor(
            np.random.randn(72, seq, d_model), dtype=torch.float32
        )
        self._labels = torch.tensor(np.random.randn(72, seq), dtype=torch.float32)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return (self._inputs[idx % 72], self._labels[idx % 72])

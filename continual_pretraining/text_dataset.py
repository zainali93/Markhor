import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Split a tokenized corpus into non-overlapping fixed-length sequences."""

    def __init__(self, tokens, block_size=512):
        self.tokens = tokens
        self.block_size = block_size
        self.num_samples = len(tokens) // block_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        return torch.tensor(
            self.tokens[start:end],
            dtype=torch.long
        ).clone().detach()

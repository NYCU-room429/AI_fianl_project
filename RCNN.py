import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class CRNNDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be overridden by subclasses.")

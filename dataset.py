from config import config

import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer

from datasets import load_dataset


config = config()

class BilingualDataset(Dataset):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.ds = get_ds(config)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


def get_ds(config):
    ds = load_dataset(config['dataset'])
    return ds['train']


if __name__ == '__main__':
    ds = BilingualDataset(config)
    print(ds.__len__(), ds.__getitem__(0), sep='\n')
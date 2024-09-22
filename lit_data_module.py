from calendar import c

import math

import torch.utils
import torch.utils.data

from dataset import BilingualDataset
from config import configuration

import torch
from torch.utils.data import DataLoader, Subset

import lightning as L
from lightning import LightningDataModule

config = configuration()

class DataModuleLightning(LightningDataModule):

    def __init__(self, data_dir='./Data'):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = BilingualDataset()

        # Declare train/val/test data fractions
        self.train_frac = config['train_frac']
        self.test_val_frac = config['test_val_frac']

        # Create train/val/test indices
        self.train_indices = [i for i in range(1, math.floor(len(self.dataset.ds['train']) * self.train_frac))]
        self.val_indices = [i for i in range(self.train_indices[-1], math.floor((self.train_frac + ((1 - self.train_frac) * self.test_val_frac)) * len(self.dataset.ds['train'])))]
        self.test_indices = [i for i in range(self.val_indices[-1], len(self.dataset.ds['train']))]

        print('\n\nTRAIN/VAL/TEST SETS:')
        print(len(self.train_indices), len(self.val_indices), len(self.test_indices), sep='\n')
        print('\n')

    def prepare_data(self):
        pass
    
    def setup(self, stage: str):
        '''
        This method is used to create subsets of dataset
        for training, validation, and testing
        '''
        if stage == 'fit' or stage is None:
            self.train_set = Subset(self.dataset, self.train_indices)
            self.val_set = Subset(self.dataset, self.val_indices)
        
        elif stage == 'test':
            self.test_set = Subset(self.dataset, self.test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=config['batch_size'])

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=config['batch_size'])
        
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=config['batch_size'])

    def predict_dataloader(self):
        pass

if __name__ == '__main__':
    pass
from calendar import c

import math

import torch.utils
import torch.utils.data
from dataset import BilingualDataset
from config import config

import torch
import lightning as L
from lightning import LightningDataModule

config = config()

class DataModuleLightning(L.LightningDataModule):

    def __init__(self, config, data_dir='./Data'):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = BilingualDataset(config)
        print(len(self.dataset))

        # Declare train/val/test data fractions
        self.train_frac = config['train_frac']
        self.test_val_frac = config['test_val_frac']

        # Create train/val/test indices
        self.train_indices = [i for i in range(1, math.floor(len(self.dataset) * self.train_frac))]
        self.val_indices = [i for i in range(self.train_indices[-1], math.floor((self.train_frac + ((1 - self.train_frac) * self.test_val_frac)) * len(self.dataset)))]
        self.test_indices = [i for i in range(self.val_indices[-1], len(self.dataset))]

    def prepare_data(self):
        pass
    
    def setup(self, stage: str):
        '''
        This method is used to create subsets of dataset
        for training, validation, and testing
        '''
        if stage == 'fit' or stage is None:
            train_set = torch.utils.data.Subset(self.dataset, self.train_indices)
            val_set = torch.utils.data.Subset(self.dataset, self.val_indices)
            pass
        
        elif stage == 'test':
            test_set = torch.utils.data.Subset(self.dataset, self.test_indices)

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass
        
    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

if __name__ == '__main__':
    '''train_indices = [i for i in range(1, math.floor(101 * 0.9))]
    val_indices = [i for i in range(train_indices[-1], math.floor((0.9 + ((1 - 0.9) * 0.5)) * 101))]
    test_indices = [i for i in range(val_indices[-1], 101)]

    print(train_indices, val_indices, test_indices, sep='\n')'''
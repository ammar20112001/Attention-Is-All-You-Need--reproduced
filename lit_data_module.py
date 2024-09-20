from calendar import c
from dataset import BilingualDataset
from config import config

import lightning as L
from lightning import LightningDataModule

config = config()

class DataModuleLightning(L.LightningDataModule):

    def __init__(self, config, data_dir='./Data'):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = BilingualDataset(config)

    def prepare_data(self):
        pass
    
    def setup(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass
        
    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

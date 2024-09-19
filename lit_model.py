import torch
import lightning as L

class transformerLightning(L.LightningModule):
    
    def __init__(self, model, config):
        super().__init__()
        self.model = model()

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self):
        pass

    def predict_step(self):
        pass

from lit_model import transformerLightning
from lit_data_module import DataModuleLightning
from config import configuration
from sweep_config import sweep_configuration

import lightning as L

import wandb

wandb.login()
# Import sweep configuration
config = configuration()
sweep_config = sweep_configuration()

# Create data instance
dataset = DataModuleLightning(config)
# Create dataloader
train_loader = DataModuleLightning.train_dataloader()

# Create model instance
model = transformerLightning(config)

# Train model
trainer = L.Trainer()
trainer.fit(model=model, train_dataloaders=train_loader)
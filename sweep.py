from lit_model import transformerLightning, checkpoint_callback, wandb_logger
from lit_data_module import DataModuleLightning
from config import configuration
from sweep_config import sweep_configuration

import lightning as L

import wandb

# Import sweep configuration
sweep_config = sweep_configuration()

wandb.login()
sweep_id = wandb.sweep(sweep_config, project="Attention-Is-All-You-Need--reproduced")

def train(config=None):
    # Initialize new wandb run
    with wandb.init(config=config):
        # if called by wandb agent, as below, 
        # this config will be set by Sweep Controller
        config = wandb.config

    # Create data instance
    dataset = DataModuleLightning(config)
    # Create dataloader
    dataset.setup(stage='fit')
    train_loader = dataset.train_dataloader()

    # Create model instance
    model = transformerLightning(config)

    # Train model
    trainer = L.Trainer(max_epochs=config.epochs, 
                        callbacks=[checkpoint_callback], 
                        logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader)

wandb.agent(sweep_id, train, count=3)
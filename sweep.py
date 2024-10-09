from lit_model import transformerLightning, checkpoint_callback
from lit_data_module import DataModuleLightning
from sweep_config import sweep_configuration

import lightning as L
from lightning.pytorch.loggers import WandbLogger

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

        # Initialize WandB logger
        wandb_logger = WandbLogger(
            project="Attention-Is-All-You-Need--reproduced",
            log_model="all",
            config=config,
        )

        # Create data instance
        dataset = DataModuleLightning(config)

        # Create model instance
        model = transformerLightning(config, logger=wandb_logger)

        # Train model
        trainer = L.Trainer(
            max_epochs=config.epochs,
            callbacks=[checkpoint_callback],
            logger=wandb_logger,
        )
        # Fit model
        trainer.fit(
            model=model,
            datamodule=dataset,
        )


wandb.agent(sweep_id, train, count=3)

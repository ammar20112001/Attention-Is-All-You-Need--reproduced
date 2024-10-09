from lit_model import transformerLightning, checkpoint_callback
from lit_data_module import DataModuleLightning

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from config import configuration

if __name__ == "__main__":

    config = configuration()

    # Create WandBLogger
    wandb_logger = WandbLogger(
        project="Attention-Is-All-You-Need--reproduced", log_model="all", config=config
    )

    # Create data instance
    dataset = DataModuleLightning(config)

    # Create model instance
    model = transformerLightning(config, logger=wandb_logger)

    # Train model
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )
    # Fit model
    trainer.fit(
        model=model,
        datamodule=dataset,
    )

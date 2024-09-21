from lit_model import transformerLightning
from lit_data_module import DataModuleLightning

import torch
import lightning as L
from lightning.pytorch.cli import LightningCLI

if __name__ == '__main__':
    cli = LightningCLI(
        model_class=transformerLightning,
        datamodule_class=DataModuleLightning # COnvert this to lighning datamodule to run
        )
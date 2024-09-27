from model import build_transformer
from dataset import tokenizer
from config import configuration

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Callbacks
checkpoint_callback = ModelCheckpoint(monitor='LOSS_VAL', mode='max')

# W&B logger
wandb_logger = WandbLogger(project="Attention-Is-All-You-Need--reproduced", log_model="all")

config = configuration()

pad_token = torch.tensor(tokenizer('<pad>')['input_ids'][1:-1])
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token, label_smoothing=0.1)

class transformerLightning(L.LightningModule):
    
    def __init__(self, config_arg=config):
        super().__init__()
        self.config = config_arg
        # Initialize transformer model
        self.transformer = build_transformer(
            d_model = self.config['d_model'],
            heads = self.config['heads'],
            n_stack = self.config['n_stack'],
            max_seq_len = self.config['max_seq_len'],
            src_vocab_size = self.config['src_vocab_size'],
            tgt_vocab_size = self.config['tgt_vocab_size'],
            dropout = self.config['dropout'],
            d_fc = self.config['d_fc']
        )
        # Log hyper-parameters
        self.save_hyperparameters()

    @staticmethod
    def check_translation(encoder_input, decoder_input, labels, log_text_len):
        output = torch.argmax(labels, dim=-1)
        data = []
        columns = ['Input', 'Target', 'Model output']

        #print('\n')
        for i in range(log_text_len):
            data.append(
                [
                  tokenizer.decode(encoder_input[i], skip_special_tokens=True),
                  tokenizer.decode(decoder_input[i], skip_special_tokens=True),
                  tokenizer.decode(output[i], skip_special_tokens=True)
                ]
            )
        return  columns, data

    def training_step(self, batch, batch_idx):
        # Extracting required inputs
        encoder_input = batch['encoder_input']
        decoder_input = batch['decoder_input']
        encoder_mask = batch['encoder_mask']
        decoder_mask = batch['decoder_mask']

        # Encoding source text
        encoder_output = self.transformer.encode(encoder_input, encoder_mask) # --> (B, S, d_model)

        # Decoding to target text
        decoder_output = self.transformer.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # --> (B, S, d_model)

        # Projecting from (B, S, d_model) to (B, S, V)
        logits = self.transformer.project(decoder_output) # --> (B, S, V)

        # Extracting labels for loss
        labels = batch['labels'] # --> (B, S)

        # Calculating Loss
        loss = loss_fn(logits.transpose(1,2), labels)

        # Logging Metrics
        self.log("LOSS", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Extracting required inputs
        encoder_input = batch['encoder_input']
        decoder_input = batch['decoder_input']
        encoder_mask = batch['encoder_mask']
        decoder_mask = batch['decoder_mask']

        # Encoding source text
        encoder_output = self.transformer.encode(encoder_input, encoder_mask) # --> (B, S, d_model)

        # Decoding to target text
        decoder_output = self.transformer.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # --> (B, S, d_model)

        # Projecting from (B, S, d_model) to (B, S, V)
        logits = self.transformer.project(decoder_output) # --> (B, S, V)

        # Extracting labels for loss
        labels = batch['labels'] # --> (B, S)

        # Calculating Loss
        loss = loss_fn(logits.transpose(1,2), labels)

        # Logging metrics and text
        self.log("LOSS_VAL", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx==0:
            columns, data = transformerLightning.check_translation(encoder_input, decoder_input, logits, self.config['log_text_len'])
            wandb_logger.log_text(key="samples", columns=columns, data=data)
    
    def test_step(self):
        pass

    def predict_step(self):
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config['lr'])

if __name__ == '__main__':
    pass
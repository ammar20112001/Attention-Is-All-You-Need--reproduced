from model import build_transformer
from dataset import tokenizer
from config import configuration

import torch
import lightning as L

config = configuration()

pad_token = torch.tensor(tokenizer('<pad>')['input_ids'][1:-1])
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token, label_smoothing=0.1)

class transformerLightning(L.LightningModule):
    
    def __init__(self):
        super().__init__()
        # Initialize transformer model
        self.transformer = build_transformer(
            d_model = config['d_model'],
            heads = config['heads'],
            n_stack = config['n_stack'],
            max_seq_len = config['max_seq_len'],
            src_vocab_size = config['src_vocab_size'],
            tgt_vocab_size = config['tgt_vocab_size'],
            dropout = config['dropout'],
            d_fc = config['d_fc']
        )
        # Log hyper-parameters
        self.save_hyperparameters()

    @staticmethod
    def check_translation(encoder_input, decoder_input, labels):
        output = torch.argmax(labels, dim=-1)
        print('\n')
        for i in range(10):
            print(f'\n\nInput:          {tokenizer.decode(encoder_input[i], skip_special_tokens=True)}',
                f'Target:         {tokenizer.decode(decoder_input[i], skip_special_tokens=True)}',
                '----------------------------------------------------------------------------',
                f'Model output:   {tokenizer.decode(output[i], skip_special_tokens=True)}',
                sep='\n')
        print('\n\n')

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

        # Logging Metrics
        self.log("LOSS_VAL", loss, on_epoch=True, prog_bar=True, logger=True)
        # Translation check
        if batch_idx==0:
            transformerLightning.check_translation(encoder_input, decoder_input, labels)
    
    def test_step(self):
        pass

    def predict_step(self):
        pass

if __name__ == '__main__':
    pass
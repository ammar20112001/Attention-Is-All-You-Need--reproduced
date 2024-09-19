from model import build_transformer

import torch
import lightning as L

class transformerLightning(L.LightningModule):
    
    def __init__(self, config):
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

    def training_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input']
        decoder_input = batch['decoder_input']
        labels = batch['labels']
        encoder_mask = batch['encoder_mask']
        decoder_mask = batch['decoder_mask']

        # Encoding source text
        encoder_output = self.transformer.encode(encoder_input, encoder_mask) # --> (B, S, d_model)

        # Decoding to target text
        decoder_output = self.transformer.encode(decoder_input, encoder_output, encoder_mask, decoder_mask) # --> (B, S, d_model)

        # Projecting from (B, S, d_model) to (B, S, Vy_hat
        logits = self.transformer.project(decoder_output)

        # Calculating Loss
        loss = torch.nn.functional.nll_loss(logits, labels.view(-1))

        return loss

    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self):
        pass

    def predict_step(self):
        pass

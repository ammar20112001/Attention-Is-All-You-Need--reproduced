from model import build_transformer
from dataset import tokenizer
from config import configuration

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

import wandb

config = configuration()

# Callbacks
checkpoint_callback = ModelCheckpoint(monitor="LOSS_VAL", mode="max")

pad_token = torch.tensor(tokenizer("<pad>")["input_ids"][1:-1])
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token, label_smoothing=0.1)


class transformerLightning(L.LightningModule):
    def __init__(self, config_arg=config, logger=None):
        super().__init__()
        self.config = config_arg
        # Initialize transformer model
        self.transformer = build_transformer(
            d_model=self.config["d_model"],
            heads=self.config["heads"],
            n_stack=self.config["n_stack"],
            max_seq_len=self.config["max_seq_len"],
            src_vocab_size=self.config["src_vocab_size"],
            tgt_vocab_size=self.config["tgt_vocab_size"],
            dropout=self.config["dropout"],
            d_fc=self.config["d_fc"],
        )

        # W&B logger
        self.wandb_logger = logger

        # W&B watch to log gradients and model topology
        # wandb_logger.watch(self.transformer, log="all", log_freq=250)

        # Store logits for histograms
        self.logits = []

        # Create special tokens for predict data transformation
        self.sos_token = torch.tensor(tokenizer("<sos>")["input_ids"][1:-1])
        self.eos_token = torch.tensor(tokenizer("<eos>")["input_ids"][1:-1])
        self.pad_token = torch.tensor(tokenizer("<pad>")["input_ids"][1:-1])

    @staticmethod
    def check_translation(encoder_input, decoder_input, labels, log_text_len):
        output = torch.argmax(labels, dim=-1)
        data = []
        columns = ["Input", "Target", "Model output"]

        # print('\n')
        for i in range(log_text_len):
            data.append(
                [
                    tokenizer.decode(encoder_input[i], skip_special_tokens=True),
                    tokenizer.decode(decoder_input[i], skip_special_tokens=True),
                    tokenizer.decode(output[i], skip_special_tokens=True),
                ]
            )
        return columns, data

    def training_step(self, batch, batch_idx):
        # Extracting required inputs
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]

        # Encoding source text
        encoder_output = self.transformer.encode(
            encoder_input, encoder_mask
        )  # --> (B, S, d_model)

        # Decoding to target text
        decoder_output = self.transformer.decode(
            decoder_input, encoder_output, encoder_mask, decoder_mask
        )  # --> (B, S, d_model)

        # Projecting from (B, S, d_model) to (B, S, V)
        logits = self.transformer.project(decoder_output)  # --> (B, S, V)

        # Extracting labels for loss
        labels = batch["labels"]  # --> (B, S)

        # Calculating Loss
        loss = loss_fn(logits.transpose(1, 2), labels)

        # Logging Metrics
        self.log("LOSS", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Extracting required inputs
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]

        # Encoding source text
        encoder_output = self.transformer.encode(
            encoder_input, encoder_mask
        )  # --> (B, S, d_model)

        # Decoding to target text
        decoder_output = self.transformer.decode(
            decoder_input, encoder_output, encoder_mask, decoder_mask
        )  # --> (B, S, d_model)

        # Projecting from (B, S, d_model) to (B, S, V)
        logits = self.transformer.project(decoder_output)  # --> (B, S, V)

        # Extracting labels for loss
        labels = batch["labels"]  # --> (B, S)

        # Calculating Loss
        loss = loss_fn(logits.transpose(1, 2), labels)

        # Logging metrics and text
        self.log(
            "LOSS_VAL", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if batch_idx == 0 and self.wandb_logger is not None:
            columns, data = transformerLightning.check_translation(
                encoder_input, decoder_input, logits, self.config["log_text_len"]
            )
            self.wandb_logger.log_text(key="samples", columns=columns, data=data)
            # Append logits for histogram logging of logits
            self.logits.append(logits.cpu())

    def on_validation_epoch_end(self) -> None:
        if self.wandb_logger is not None:
            # Log histogram of logits
            flattened_logits = torch.flatten(torch.cat(self.logits))
            self.logger.experiment.log(
                {
                    "valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
                    "global_step": self.global_step,
                }
            )
            self.logits = []  # Reset logits for next validation run

    def test_step(self):
        pass

    def predict(self, x):

        # Tokenize sentence
        ds_src_tokens = tokenizer(
            x,
            add_special_tokens=False,
            truncation=True,
            max_length=config["dec_max_seq_len"],
        )["input_ids"]
        ds_tgt_tokens = []

        # Length of padding tokens in encoder and decoder inputs
        enc_num_pad_tokens = (
            config["enc_max_seq_len"] - len(ds_src_tokens) - 2
        )  # (-) <sos> and <eos>
        dec_num_pad_tokens = (
            config["dec_max_seq_len"] - len(ds_tgt_tokens) - 1
        )  # (-) <sos> in decoder input

        # Create encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(ds_src_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.int64)
                # <sos> ...sentence tokens... <eos> <pad>...
            ],
            dim=0,
        )

        # Create decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(ds_tgt_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
                # <sos> ...sentence tokens... <pad>...
            ],
            dim=0,
        )

        # Create encoder mask
        encoder_mask = (
            (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        )  # (1, 1, seq_len)

        # Encoding source text
        encoder_output = self.transformer.encode(
            encoder_input, encoder_mask
        )  # --> (B, S, d_model)

        token_num = 0
        while len(ds_tgt_tokens) + 1 < config["dec_max_seq_len"]:

            # Create decoder mask
            decoder_mask = (decoder_input != self.pad_token).unsqueeze(
                0
            ).int() & transformerLightning.causal_mask(
                decoder_input.size(0)
            )  # (1, seq_len) & (1, seq_len, seq_len)

            encoder_input = encoder_input.reshape((1, -1))
            decoder_input = decoder_input.reshape((1, -1))

            # Decoding to target text
            decoder_output = self.transformer.decode(
                decoder_input, encoder_output, encoder_mask, decoder_mask
            )  # --> (B, S, d_model)

            # Projecting from (B, S, d_model) to (B, S, V)
            logits = self.transformer.project(decoder_output)  # --> (B, S, V)

            # Find output token
            output_token = torch.argmax(logits, dim=-1).tolist()[0][
                token_num
            ]  # --> (B, S)
            ds_tgt_tokens.append(output_token)

            dec_num_pad_tokens = (
                config["dec_max_seq_len"] - len(ds_tgt_tokens) - 1
            )  # (-) <sos> in decoder input

            # Update decoder input
            decoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(ds_tgt_tokens, dtype=torch.int64),
                    torch.tensor(
                        [self.pad_token] * dec_num_pad_tokens, dtype=torch.int64
                    )
                    # <sos> ...sentence tokens... <pad>...
                ],
                dim=0,
            )

            if output_token == self.eos_token:
                break

            token_num += 1

        return decoder_input

    @staticmethod
    def causal_mask(size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if config["optimizer"] == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        elif config["optimizer"] == "SGD":
            return torch.optim.SGD(self.parameters(), lr=self.config["lr"])


if __name__ == "__main__":
    pass

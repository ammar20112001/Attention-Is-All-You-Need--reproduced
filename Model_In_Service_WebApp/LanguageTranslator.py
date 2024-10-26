"""Translates a text from English to French.
"""

import argparse

import torch

from transformers import AutoTokenizer


MODEL_DIRECTORY = "../Models"
MODEL_NAME = "model.pt"


class LanguageTranslator:
    def __init__(self, path: str = None):

        # TorchScript model path
        if not path:
            self.model_path = MODEL_DIRECTORY + "/" + MODEL_NAME
        else:
            self.model_path = path

        self.model = torch.jit.load(self.model_path)

        # Initializing tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        special_tokens_dict = {
            "bos_token": "<sos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def predict(self, x: str):

        # Tokenize sentence x
        x = torch.tensor(
            self.tokenizer(
                x,
                add_special_tokens=False,
                truncation=True,
                max_length=256,
            )["input_ids"],
            dtype=torch.int64,
        )

        # Predict tokens
        y = self.model(x)

        # return decoded tokens
        return self.tokenizer.decode(token_ids=y, skip_special_tokens=True)


def _setup_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    parser.add_argument(
        "sentence",
        type=str,
        help="English sentence that will be translated to French",
    )

    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="Model file path to use",
    )

    return parser


def main():
    # Parse CLI arguments
    parser = _setup_parser()
    args = parser.parse_args()

    # Create model class
    model = LanguageTranslator(args.filename)

    # Predict
    pred = model.predict(args.sentence)
    print(pred)


if __name__ == "__main__":
    main()

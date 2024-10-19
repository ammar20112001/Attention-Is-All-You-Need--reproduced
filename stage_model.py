import torch

from lit_model import transformerLightning

import argparse

import wandb

# Local directory to download model
LOCAL_MODEL_DIRECTORY = "models"
LOCAL_PROD_MODEL_DIRECTORY = "prod/models"

# Stage a model that has been trained and stored as an artifcat on W&B


# Download model to production directory
# Download model to W&B for other uses


def main(args):

    # Login W&B
    wandb.login()
    api = wandb.Api()
    # Get api run
    run = api.run(f"{args.entity}/{args.from_project}/{args.artifact}")

    # Load model artifact
    artifact = api.artifact(
        f"{args.entity}/{args.from_project}/model-{args.artifact}:{args.version}"
    )
    # Downlaod model artifact
    artifact.download(root=LOCAL_MODEL_DIRECTORY)

    # Convert PyTorch model to TorchScript for production environment
    model = transformerLightning.load_from_checkpoint(
        f"{LOCAL_MODEL_DIRECTORY}/model.ckpt"
    )
    scripted_module = model.to_torchscript()

    # Save for use in production environment
    torch.jit.save(scripted_module, f"{LOCAL_PROD_MODEL_DIRECTORY}/model.pt")
    # Also save TorchScript version to W&B for other accesses
    run.upload_file(f"{LOCAL_PROD_MODEL_DIRECTORY}/model.pt")


def _setup_parser():
    parser = argparse.ArgumentParser(
        prog="Stage Model", description="Add necessary argument to stage a model"
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        default=False,
        help="True to stage model, and False otherwise",
    )
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--from_project", type=str, default=None)
    parser.add_argument("--artifact", type=str, default=None)
    parser.add_argument("--version", type=str, default="latest")

    parser.add_argument("--torchscript", action="store_true", default=False)
    parser.add_argument("--sentence", type=str, default="English")

    return parser


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()

    if args.fetch:
        main(args)

    elif not args.fetch and not args.torchscript:
        print("Using lightning model...")
        from dataset import tokenizer

        model = transformerLightning.load_from_checkpoint("models/model.ckpt")

        sentence = torch.tensor(
            tokenizer(
                args.sentence,
                add_special_tokens=False,
                truncation=True,
                max_length=256,
            )["input_ids"],
            dtype=torch.int64,
        )

        print("English:", args.sentence)
        y = model(sentence)
        decoded = tokenizer.decode(token_ids=y, skip_special_tokens=True)
        print("French:", decoded)

    elif not args.fetch and args.torchscript:
        print("Using TorchScript model.pt...")
        from dataset import tokenizer

        model = torch.jit.load("prod/models/model.pt")

        sentence = torch.tensor(
            tokenizer(
                args.sentence,
                add_special_tokens=False,
                truncation=True,
                max_length=256,
            )["input_ids"],
            dtype=torch.int64,
        )

        print("English:", args.sentence)
        y = model(sentence)
        decoded = tokenizer.decode(token_ids=y, skip_special_tokens=True)
        print("French:", decoded)

import torch

from dataset import tokenizer

from lit_model import transformerLightning

import argparse

import wandb

# Local directory to download model
LOCAL_ARTIFACT_DIRECTORY = "prod/model/"

# Stage a model that has been trained and stored as an artifcat on W&B


# Download model to production directory
# Download model to W&B for other uses


def main(args):

    # Login W&B
    wandb.login()
    api = wandb.Api()

    # Load and download model and metadata from W&B
    run = api.run(f"{args.entity}/{args.from_project}/{args.artifact}")

    artifact = api.artifact(
        f"{args.entity}/{args.from_project}/model-{args.artifact}:{args.version}"
    )
    artifact.download(root=LOCAL_ARTIFACT_DIRECTORY)

    # Convert PyTorch model to torchscript for production environment
    model = transformerLightning(config_arg=run.config)
    scripted_module = model.to_torchscript()

    # Sace for use in production environment
    torch.jit.save(scripted_module, f"{LOCAL_ARTIFACT_DIRECTORY}/model.pt")


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
    parser.add_argument("--sentence", type=str, default="english")

    return parser


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()

    if args.fetch:
        main(args)

    elif not args.fetch:
        model = transformerLightning.load_from_checkpoint("prod/model.ckpt")
        print("English:", args.sentence)
        y_hat = model.predict(args.sentence)
        decoded = tokenizer.decode(token_ids=y_hat, skip_special_tokens=True)
        print("French: ", decoded)

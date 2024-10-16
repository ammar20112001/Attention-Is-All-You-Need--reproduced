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
    torch.jit.save(
        scripted_module, f"{LOCAL_PROD_MODEL_DIRECTORY}/model-{args.artifact}.pt"
    )
    # Also save TorchScript version to W&B for other accesses
    run.upload_file(f"{LOCAL_PROD_MODEL_DIRECTORY}/model-{args.artifact}.pt")


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

    return parser


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()

    if args.fetch:
        main(args)

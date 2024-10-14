import argparse

import wandb

# Local directtory to download model
LOCAL_ARTIFACT_DIRECTORY = "prod/"

# Stage a model that has been trained and stored as an artifcat on W&B


# Download model to production directory
# Download model to W&B for other uses


def main(args):

    # Login W&B
    wandb.login()
    api = wandb.Api()
    # Load model from W&B
    artifact = api.artifact(f"{args.entity}/{args.from_project}/{args.artifact}:latest")
    artifact.download(root=LOCAL_ARTIFACT_DIRECTORY)

    # Convert PyTorch model to torchscript for production environment


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

    return parser


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()
    main(args)

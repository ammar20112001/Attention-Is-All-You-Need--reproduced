import os
import sys
import zipfile

import boto3

import json

from LanguageTranslator import LanguageTranslator


s3 = boto3.client("s3")


def load_torch(bucket, key):

    torch_dir = "/tmp/torch"

    if not os.path.exists(torch_dir):
        # Download model from s3 to temp dir
        s3.Bucket(bucket).download_file(key, f"{torch_dir}.zip")
        # Unzip Torch dependency to /tmp/ directory
        zipfile.ZipFile(torch_dir, "r").extractall("./")
        # Delete zipped torch file
        os.remove(f"{torch_dir}.zip")

    # Add /tmp/torch to environment
    sys.path.append(torch_dir)


def download_model(bucket, key):
    model_dir = f"/tmp/{os.path.basename(key)}"

    if not os.path.exists(model_dir):
        # Download model from s3 to temp dir
        s3.Bucket(bucket).download_file(key, model_dir)

    return model_dir


def load_model(model_dir):
    return LanguageTranslator(path=model_dir)


def lambda_handler(event, context):
    x = json.loads(event["body"])["englishSentence"]

    # Load torch in /tmp/
    load_torch(bucket="translator-model-bucket", key="torch.zip")

    # Downloads model from s3 bucket and return path
    model_dir = download_model(bucket="translator-model-bucket", key="model.pt")

    # Load model
    model = load_model(model_dir)

    # Make prediction
    y = model(x)

    return {"statusCode": 200, "prediction": y}

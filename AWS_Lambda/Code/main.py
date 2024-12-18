import os
import sys
import zipfile

import json

import boto3


os.environ["HF_HOME"] = "/tmp"

s3 = boto3.resource("s3")


def load_torch(bucket, key):

    torch_dir = "/tmp/torch/"

    if not os.path.exists(torch_dir):
        # Download model from s3 to temp dir
        s3.Bucket(bucket).download_file(key, "/tmp/torch.zip")
        # Unzip Torch dependency to /tmp/ directory
        zipfile.ZipFile("/tmp/torch.zip", "r").extractall("/tmp/")
        # Delete zipped torch file

    if os.path.exists("/tmp/torch.zip"):
        os.remove("/tmp/torch.zip")

    # Add /tmp/torch to environment
    sys.path.append("/var/task/tmp/")
    sys.path.append("/tmp/")


def download_model(bucket, key):
    model_dir = f"/tmp/{os.path.basename(key)}"

    if not os.path.exists(model_dir):
        # Download model from s3 to temp dir
        s3.Bucket(bucket).download_file(key, model_dir)

    return model_dir


def load_model(model_dir):
    from LanguageTranslator import LanguageTranslator

    return LanguageTranslator(path=model_dir)


def lambda_handler(event, context):
    body = json.loads(event["body"])
    x = body.get("englishSentence")

    # Load torch in /tmp/
    load_torch(bucket="translator-model-bucket", key="torch.zip")

    # Downloads model from s3 bucket and return path
    model_dir = download_model(bucket="translator-model-bucket", key="model.pt")

    # Load model
    model = load_model(model_dir)

    # Make prediction
    y = model.predict(x)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"message": "Data received successfully!", "prediction": y}),
    }

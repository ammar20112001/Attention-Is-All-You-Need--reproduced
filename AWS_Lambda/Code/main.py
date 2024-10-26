import os

import boto3

import json

from LanguageTranslator import LanguageTranslator

try:
    import unzip_torch
except Exception as e:
    print("Failed at unzip_torch", e, sep="\n")


s3 = boto3.client('s3')


def download_model(bucket, object):
    model_dir = f"/tmp/{os.path.basename(object)}"

    if not os.path.exists(model_dir):
        # Download model from s3 to temp dir
        s3.download_file(bucket, object, model_dir)
    
    return model_dir


def load_model(model_dir):
    return LanguageTranslator(path=model_dir)


def lambda_handler(event, context):
    x = json.loads(event["body"])["englishSentence"]

    # Downloads model from s3 bucket and return path
    model_dir = download_model(
        bucket='',
        object=''
        )
    
    # Load model
    model = load_model(model_dir)

    # Make prediction
    y = model(x)

    return {
        "status": 200,
        "prediction": y
    }

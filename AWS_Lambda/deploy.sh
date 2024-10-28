#!/bin/bash

# Create a packages folder and copy everything from .venv
mkdir packages
cp -r .venv/lib64/python3.12/site-packages/* packages

cd packages

# Remove unnecessary dependencies for production
find . -type d -name "tests" -exec rm -rf {} +
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf ./{caffe2,wheel,wheel-*,pkg_resources,boto*,aws*,pip,pip-*,pipenv,setuptools}
rm -rf ./{*.egg-info,*.dist-info}
find . -name \*.pyc -delete

# Zip and delete torch
zip -r9 ${OLDPWD}/torch.zip torch
rm -r torch

# Zip and delete all other dependencies
zip -r9 ${OLDPWD}/dependencies.zip .
cd ${OLDPWD}
rm -r packages

# Zip all code, adding dependencies zipped content within
cd Code
zip -ru ../dependencies.zip .
cd ..

# Upload zipped file to s3 bucket
aws s3 cp dependencies.zip s3://translator-model-bucket/
aws s3 cp torch.zip s3://translator-model-bucket/

# Upload to aws lambda from s3 bucket
aws lambda update-function-code --function-name arn:aws:lambda:eu-north-1:257394486576:function:TranformersTranslator --s3-bucket translator-model-bucket --s3-key dependencies.zip
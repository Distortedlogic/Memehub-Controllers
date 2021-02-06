import json
import os
from pathlib import Path

import boto3
import ml2rt
import redisai
from botocore.exceptions import ClientError
from controller.constants import MODEL_REPO, STATIC_PATH
from decouple import config
from tqdm import tqdm

s3 = boto3.client(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY"),
)


def download_aws(path):
    if not Path(path).is_file():
        with open(path, "wb") as f:
            try:
                s3.download_fileobj("memehub", "memehub/models/" + path, f)
            except ClientError:
                pass


def download_from_aws():
    download_aws(STATIC_PATH)
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)
    for name in static["names"]:
        download_aws(MODEL_REPO + f"{name}.pt")
    return static


def load_base_to_redisai(device="CPU", backend="torch"):
    download_aws(MODEL_REPO + "features.pt")
    download_aws(MODEL_REPO + "MemeClf.pt")
    rai = redisai.Client(host="redis", port="6379")
    model = ml2rt.load_model(MODEL_REPO + f"features.pt")
    rai.modelset("features", backend, device, model)
    model = ml2rt.load_model(MODEL_REPO + f"MemeClf.pt")
    rai.modelset("MemeClf", backend, device, model)
    print("Base Loaded")


def load_stonks_to_redisai(device="CPU", backend="torch"):
    static = download_from_aws()
    print("AWS Download Complete")
    rai = redisai.Client(host="redis", port="6379")
    print("loaded_models", rai.modelscan())
    names_to_load = [
        name
        for name in static["names"]
        if name not in [data[0] for data in rai.modelscan()]
    ]
    for name in tqdm(names_to_load, total=len(names_to_load)):
        model = ml2rt.load_model(MODEL_REPO + f"{name}.pt")
        rai.modelset(name, backend, device, model)

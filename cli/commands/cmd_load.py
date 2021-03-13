import json
from pathlib import Path
from typing import Any, cast

import boto3
import click
import ml2rt
import redisai
from botocore.exceptions import ClientError
from decouple import config

VERSION = "0.2.3"

MODEL_REPO = f"{VERSION}/jit/"
Path(MODEL_REPO).mkdir(parents=True, exist_ok=True)
STATIC_PATH = f"{VERSION}/static.json"

s3: Any = boto3.client(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY"),
)

device = "CPU"
backend = "torch"


def download_aws(path: str):
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


@click.group()
def cli():
    """ Run Imgflip Related Scripts"""
    pass


@click.command()
def base():
    """
    Load models into redisai
    """
    download_aws(MODEL_REPO + "features.pt")
    download_aws(MODEL_REPO + "MemeClf.pt")
    download_aws(MODEL_REPO + "dense.pt")
    rai = redisai.Client(host="redis", port=6379)
    model = ml2rt.load_model(MODEL_REPO + f"features.pt")
    _ = rai.modelset(
        "features",
        backend,
        device,
        cast(Any, model),
        tag=VERSION,
        inputs=cast(Any, None),
        outputs=cast(Any, None),
    )
    model = ml2rt.load_model(MODEL_REPO + f"MemeClf.pt")
    _ = rai.modelset(
        "MemeClf",
        backend,
        device,
        cast(Any, model),
        tag=VERSION,
        inputs=cast(Any, None),
        outputs=cast(Any, None),
    )
    model = ml2rt.load_model(MODEL_REPO + f"dense.pt")
    _ = rai.modelset(
        "dense",
        backend,
        device,
        cast(Any, model),
        tag=VERSION,
        inputs=cast(Any, None),
        outputs=cast(Any, None),
    )
    print("Base Loaded")

    return None


@click.command()
def stonks():
    """
    Load models into redisai
    """
    static = download_from_aws()
    rai = redisai.Client(host="redis", port=6379)
    current_stonks = [data[0] for data in rai.modelscan()]
    names_to_load = [name for name in static["names"] if name not in current_stonks]
    for idx, name in enumerate(names_to_load):
        print(f"{idx}/{len(names_to_load)} - {name}")
        model = ml2rt.load_model(MODEL_REPO + f"{name}.pt")
        _ = rai.modelset(
            name,
            backend,
            device,
            cast(Any, model),
            tag=VERSION,
            inputs=cast(Any, None),
            outputs=cast(Any, None),
        )
    print("Stonks Loaded")

    return None


cli.add_command(base)
cli.add_command(stonks)

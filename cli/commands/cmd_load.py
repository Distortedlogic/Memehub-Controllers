import os
from pathlib import Path
from typing import Any, cast

import boto3
import click
import ml2rt
import redisai
from botocore.exceptions import ClientError
from decouple import config
from src.constants import LOAD_VERSION, MODELS_REPO

JIT_REPO = MODELS_REPO + f"{LOAD_VERSION}/jit/"
Path(JIT_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore

s3: Any = boto3.client(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY"),
)
s3 = boto3.resource(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY")
)
bucket = s3.Bucket("memehub")

device = "CPU"
backend = "torch"


def download_aws(local_repo: str, path: str):
    if not Path(local_repo + path).is_file():
        with open(local_repo + path, "wb") as f:
            try:
                s3.download_fileobj("memehub", "memehub/models/" + path, f)
            except ClientError:
                pass


@click.group()
def cli():
    """ Run Imgflip Related Scripts"""
    pass


@click.command()
def stonk_market():
    """
    Load models into redisai
    """
    for name in list(bucket.objects.filter(Prefix=f"memehub/models/{LOAD_VERSION}")):
        download_aws(JIT_REPO, f"{name}.pt")
    rai = redisai.Client(host="redis", port=6379)
    current_stonks = [data[0] for data in rai.modelscan()]
    names_on_disk = [
        os.path.splitext(file)[0]
        for file in os.listdir(MODELS_REPO + f"/{LOAD_VERSION}/jit/")
        if "backup" not in file
    ]
    names_to_load = [name for name in names_on_disk if name not in current_stonks]
    for idx, name in enumerate(names_to_load):
        print(f"{idx}/{len(names_to_load)} - {name}")
        model = ml2rt.load_model(JIT_REPO + f"{name}.pt")
        _ = rai.modelset(
            name,
            backend,
            device,
            cast(Any, model),
            tag=LOAD_VERSION,
            inputs=cast(Any, None),
            outputs=cast(Any, None),
        )
    print("Stonks Loaded")


cli.add_command(stonk_market)

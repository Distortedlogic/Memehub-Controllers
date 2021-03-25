import os
from pathlib import Path
from typing import Any, List, Tuple, cast

import boto3
import click
import ml2rt
import redisai
from botocore.exceptions import ClientError
from decouple import config
from src.constants import (
    LOAD_MEME_CLF_VERSION,
    LOAD_STONK_VERSION,
    MEME_CLF_REPO,
    STONK_REPO,
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
    for name in list(
        bucket.objects.filter(Prefix=STONK_REPO.format("jit").replace("src", "memehub"))
    ):
        download_aws(STONK_REPO.format("jit"), f"{name}.pt")
    rai = redisai.Client(host="redis", port=6379)
    current_stonks: List[str] = []
    for name, tag in cast(Tuple[str, str], rai.modelscan()):
        if tag in [LOAD_MEME_CLF_VERSION, LOAD_STONK_VERSION]:
            _ = rai.modeldel(name)
        else:
            current_stonks.append(name)
    for idx, name in enumerate(
        name
        for name in (
            os.path.splitext(file)[0]
            for file in os.listdir(MEME_CLF_REPO.format("jit"))
            if "backup" not in file
        )
        if name not in current_stonks
    ):
        model = ml2rt.load_model(MEME_CLF_REPO.format("jit") + f"{name}.pt")
        _ = rai.modelset(
            name,
            backend,
            device,
            cast(Any, model),
            tag=LOAD_MEME_CLF_VERSION,
            inputs=cast(Any, None),
            outputs=cast(Any, None),
        )
        print(f"{name} Loaded")
    names_on_disk = [
        os.path.splitext(file)[0]
        for file in os.listdir(STONK_REPO.format("jit"))
        if "backup" not in file
    ]
    names_to_load = [name for name in names_on_disk if name not in current_stonks]
    for idx, name in enumerate(names_to_load):
        print(f"{idx+1}/{len(names_to_load)} - {name}")
        model = ml2rt.load_model(STONK_REPO.format("jit") + f"{name}.pt")
        _ = rai.modelset(
            name,
            backend,
            device,
            cast(Any, model),
            tag=LOAD_STONK_VERSION,
            inputs=cast(Any, None),
            outputs=cast(Any, None),
        )


cli.add_command(stonk_market)

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
    LOAD_MEME_CLF_REPO,
    LOAD_MEME_CLF_VERSION,
    LOAD_STONK_REPO,
    LOAD_STONK_VERSION,
)

s3 = boto3.resource(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY")
)
bucket: Any = s3.Bucket("memehub")

device = "CPU"
backend = "torch"


def download_aws(path: str):
    if not Path(path).is_file():
        with open(path, "wb") as f:
            try:
                s3.download_fileobj("memehub", path.replace("src", "memehub"), f)
            except ClientError:
                pass


def download_jit_repo(repo: str):
    for name in list(
        bucket.objects.filter(Prefix=repo.format("jit").replace("src", "memehub"))
    ):
        download_aws(repo.format("jit") + f"{name}.pt")


@click.group()
def cli():
    """ Run Imgflip Related Scripts"""
    pass


@click.command()
def stonk_market():
    """
    Load Stonk models into redisai
    """
    download_jit_repo(LOAD_MEME_CLF_REPO)
    download_jit_repo(LOAD_STONK_REPO)
    rai = redisai.Client(host="redis", port=6379)
    current_stonks: List[str] = []
    for name, tag in cast(Tuple[str, str], rai.modelscan()):
        if tag in [LOAD_MEME_CLF_VERSION, LOAD_STONK_VERSION]:
            current_stonks.append(name)
        else:
            _ = rai.modeldel(name)
    for idx, name in enumerate(
        name
        for name in (
            os.path.splitext(file)[0]
            for file in os.listdir(LOAD_MEME_CLF_REPO.format("jit"))
            if "backup" not in file
        )
        if name not in current_stonks
    ):
        model = ml2rt.load_model(LOAD_MEME_CLF_REPO.format("jit") + f"{name}.pt")
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
        for file in os.listdir(LOAD_STONK_REPO.format("jit"))
        if "backup" not in file
    ]
    names_to_load = [name for name in names_on_disk if name not in current_stonks]
    for idx, name in enumerate(names_to_load):
        print(f"{idx+1}/{len(names_to_load)} - {name}")
        model = ml2rt.load_model(LOAD_STONK_REPO.format("jit") + f"{name}.pt")
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

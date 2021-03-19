from pathlib import Path
from typing import Any, cast

import boto3
import click
import ml2rt
import redisai
from botocore.exceptions import ClientError
from decouple import config
from src.constants import LOAD_VERSION, MODELS_REPO
from src.utils.model_func import get_static_names

JIT_REPO = MODELS_REPO + f"{LOAD_VERSION}/jit/"
Path(JIT_REPO).mkdir(parents=True, exist_ok=True)  # type: ignore

s3: Any = boto3.client(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY"),
)

device = "CPU"
backend = "torch"


def download_aws(local_repo: str, path: str):
    if not Path(local_repo + path).is_file():
        with open(local_repo + path, "wb") as f:
            try:
                s3.download_fileobj("memehub", "memehub/models/" + path, f)
            except ClientError:
                pass


def download_stonks_from_aws():
    download_aws(MODELS_REPO, f"{LOAD_VERSION}/static.json")
    static = get_static_names(LOAD_VERSION)
    for name in static["names"]:
        download_aws(JIT_REPO, f"{name}.pt")
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
    download_aws(JIT_REPO, "features.pt")
    download_aws(JIT_REPO, "MemeClf.pt")
    download_aws(JIT_REPO, "dense.pt")
    rai = redisai.Client(host="redis", port=6379)
    model = ml2rt.load_model(JIT_REPO + f"features.pt")
    _ = rai.modelset(
        "features",
        backend,
        device,
        cast(Any, model),
        tag=LOAD_VERSION,
        inputs=cast(Any, None),
        outputs=cast(Any, None),
    )
    model = ml2rt.load_model(JIT_REPO + f"meme_clf.pt")
    _ = rai.modelset(
        "MemeClf",
        backend,
        device,
        cast(Any, model),
        tag=LOAD_VERSION,
        inputs=cast(Any, None),
        outputs=cast(Any, None),
    )
    model = ml2rt.load_model(JIT_REPO + f"dense.pt")
    _ = rai.modelset(
        "dense",
        backend,
        device,
        cast(Any, model),
        tag=LOAD_VERSION,
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
    static = download_stonks_from_aws()
    rai = redisai.Client(host="redis", port=6379)
    current_stonks = [data[0] for data in rai.modelscan()]
    names_to_load = [name for name in static["names"] if name not in current_stonks]
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

    return None


cli.add_command(base)
cli.add_command(stonks)

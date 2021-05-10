import os
from typing import Any, List, Tuple, cast

import click
import ml2rt
import redisai
from src.constants import (
    LOAD_MEME_CLF_REPO,
    LOAD_MEME_CLF_VERSION,
    LOAD_STONK_REPO,
    LOAD_STONK_VERSION,
)

device = "CPU"
backend = "torch"


@click.group()
def cli():
    """ Run Imgflip Related Scripts"""
    pass


@click.command()
def stonk_market():
    """
    Load Stonk models into redisai
    """
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

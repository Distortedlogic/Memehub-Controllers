import json
import os
from copy import deepcopy
from typing import Dict, Iterator, List, TypedDict, cast

import numpy as np
import torch
from sqlalchemy.sql.elements import ClauseElement
from sqlalchemy.sql.functions import func
from src.constants import (
    LOAD_STATIC_PATH,
    MEME_CLF_VERSION,
    MEMES_REPO,
    MODELS_REPO,
    NOT_MEME_REPO,
    NOT_TEMPLATE_REPO,
    STONK_VERSION,
    backup,
)
from src.schema import (
    MemeCorrectTest,
    MemeCorrectTrain,
    MemeIncorrectTest,
    MemeIncorrectTrain,
    NotAMemeTest,
    NotAMemeTrain,
    NotATemplateTest,
    NotATemplateTrain,
)
from src.session import training_db
from torch import cuda

device = torch.device("cuda:0" if cuda.is_available() else "cpu")


class Smd(TypedDict):

    name_acc: Dict[str, float]
    total_time: int


def dump_smd(smd: Smd):
    with open(
        MODELS_REPO
        + "market/"
        + MEME_CLF_VERSION
        + "/stonks/"
        + STONK_VERSION
        + "/smd.json",
        "w",
    ) as f:
        json.dump(smd, f, indent=4)
    with open(
        backup(MODELS_REPO)
        + "market/"
        + MEME_CLF_VERSION
        + "/stonks/"
        + STONK_VERSION
        + "/smd.json",
        "w",
    ) as f:
        json.dump(smd, f, indent=4)


def init_smd() -> Smd:
    return {"name_acc": {}, "total_time": 0}


def get_smd(fresh: bool) -> Smd:
    if not fresh:
        try:
            with open(
                MODELS_REPO
                + "market/"
                + MEME_CLF_VERSION
                + "/stonks/"
                + STONK_VERSION
                + "/smd.json",
                "r",
            ) as f:
                smd = json.load(f)
        except Exception:
            with open(
                backup(MODELS_REPO)
                + "market/"
                + MEME_CLF_VERSION
                + "/stonks/"
                + STONK_VERSION
                + "/smd.json",
                "r",
            ) as f:
                smd = json.load(f)
        init = init_smd()
        for prop in init.keys():
            try:
                smd[prop]
            except Exception:
                smd[prop] = init[prop]
    else:
        smd = init_smd()
    return smd


class NameToMax(TypedDict):
    not_a_meme: int
    not_a_template: int
    correct: Dict[str, int]
    incorrect: Dict[str, int]


class TestTrainToMax(TypedDict):
    train: NameToMax
    test: NameToMax


class Static(TypedDict):
    names: List[str]
    names_to_shuffle: List[str]
    name_num: Dict[str, int]
    num_name: Dict[str, str]
    folder_count: Dict[str, int]
    max_name_idx: TestTrainToMax


def init_static() -> Static:
    names = [
        name
        for name, in cast(
            Iterator[str],
            training_db.query(MemeCorrectTrain.name).distinct(MemeCorrectTrain.name),
        )
    ]
    names_to_shuffle = deepcopy(names)
    name_num = {name: idx for idx, name in enumerate(names)}
    num_name = {str(v): k for k, v in name_num.items()}
    max_name_idx: TestTrainToMax = {
        "train": {
            "not_a_meme": cast(
                int, training_db.query(func.max(NotAMemeTrain.name_idx)).scalar(),
            ),
            "not_a_template": cast(
                int, training_db.query(func.max(NotATemplateTrain.name_idx)).scalar(),
            ),
            "correct": {
                name: cast(
                    int,
                    training_db.query(func.max(MemeCorrectTrain.name_idx))
                    .filter(cast(ClauseElement, MemeCorrectTrain.name == name))
                    .scalar(),
                )
                for name in names
                if name not in ["not_a_meme", "not_a_template"]
            },
            "incorrect": {
                name: cast(
                    int,
                    training_db.query(func.max(MemeIncorrectTrain.name_idx))
                    .filter(cast(ClauseElement, MemeIncorrectTrain.name == name))
                    .scalar(),
                )
                for name in names
                if name not in ["not_a_meme", "not_a_template"]
            },
        },
        "test": {
            "not_a_meme": cast(
                int, training_db.query(func.max(NotAMemeTest.name_idx)).scalar(),
            ),
            "not_a_template": cast(
                int, training_db.query(func.max(NotATemplateTest.name_idx)).scalar(),
            ),
            "correct": {
                name: cast(
                    int,
                    training_db.query(func.max(MemeCorrectTest.name_idx))
                    .filter(cast(ClauseElement, MemeCorrectTest.name == name))
                    .scalar(),
                )
                for name in names
                if name not in ["not_a_meme", "not_a_template"]
            },
            "incorrect": {
                name: cast(
                    int,
                    training_db.query(func.max(MemeIncorrectTest.name_idx))
                    .filter(cast(ClauseElement, MemeIncorrectTest.name == name))
                    .scalar(),
                )
                for name in names
                if name not in ["not_a_meme", "not_a_template"]
            },
        },
    }
    static: Static = {
        "names": names,
        "names_to_shuffle": names_to_shuffle,
        "name_num": name_num,
        "num_name": num_name,
        "folder_count": {
            "not_a_meme": len(os.listdir(NOT_MEME_REPO)),
            "not_a_template": len(os.listdir(NOT_TEMPLATE_REPO)),
            **{
                name: len(os.listdir(MEMES_REPO + name))
                for name in os.listdir(MEMES_REPO)
            },
        },
        "max_name_idx": max_name_idx,
    }
    return static


def get_static_names(version: str) -> Static:
    try:
        with open(LOAD_STATIC_PATH.format(version) + "static.json", "r") as f:
            static = json.load(f)
    except Exception:
        try:
            with open(
                backup(LOAD_STATIC_PATH.format(version)) + "static.json", "r"
            ) as f:
                static = json.load(f)
        except Exception:
            static = init_static()
    init = init_static()
    for prop in init.keys():
        try:
            static[prop]
        except Exception:
            static[prop] = init[prop]
    with open(LOAD_STATIC_PATH.format(version) + "static.json", "w") as f:
        json.dump(static, f, indent=4)
    with open(backup(LOAD_STATIC_PATH.format(version)) + "static.json", "w") as f:
        json.dump(static, f, indent=4)
    return static


class CP(TypedDict):
    iteration: int
    total_time: int
    max_acc: float
    max_val_acc: float
    max_patience: int
    min_loss: float
    loss_history: List[float]
    acc_history: List[float]
    val_acc_history: List[float]


def init_cp() -> CP:

    cp: CP = {
        "iteration": 0,
        "total_time": 0,
        "max_acc": 0,
        "max_val_acc": 0,
        "max_patience": 0,
        "min_loss": np.inf,
        "loss_history": [],
        "acc_history": [],
        "val_acc_history": [],
    }
    return cp


def load_cp(cp_path: str, fresh: bool) -> CP:
    if not fresh:
        try:
            with open(cp_path + f".json", "r") as f:
                cp = json.load(f)
        except Exception:
            try:
                with open(backup(cp_path) + f".json", "r") as f:
                    cp = json.load(f)
            except Exception:
                cp = init_cp()
        init = init_cp()
        for prop in init.keys():
            try:
                cp[prop]
            except Exception:
                cp[prop] = init[prop]
    else:
        cp = init_cp()
    return cp


def avg_n(listy: List[float], avg: int) -> List[float]:
    return [np.mean(listy[i : i + avg]) for i in range(0, len(listy), avg)]

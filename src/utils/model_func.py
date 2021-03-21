import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List, TypedDict, cast

import numpy as np
import torch
from sqlalchemy.sql.elements import ClauseElement
from sqlalchemy.sql.functions import func
from src.constants import MEMES_REPO, MODELS_REPO, NOT_MEME_REPO, NOT_TEMPLATE_REPO
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
from torch import cuda, jit, nn
from torch._C import ScriptModule

device = torch.device("cuda:0" if cuda.is_available() else "cpu")


class Smd(TypedDict):
    version: str
    name_acc: Dict[str, float]
    total_time: int


def dump_smd(smd: Smd):
    with open(MODELS_REPO + f"{smd['version']}/smd.json", "w") as f:
        json.dump(smd, f, indent=4)
    with open(MODELS_REPO + f"{smd['version']}/smd_backup.json", "w") as f:
        json.dump(smd, f, indent=4)


def init_smd(version: str) -> Smd:
    return {"version": version, "name_acc": {}, "total_time": 0}


def get_smd(fresh: bool, version: str) -> Smd:
    if not fresh:
        try:
            with open(MODELS_REPO + f"{version}/smd.json", "r") as f:
                smd = json.load(f)
        except:
            with open(MODELS_REPO + f"{version}/smd_backup.json", "r") as f:
                smd = json.load(f)
        init = init_smd(version)
        for prop in init.keys():
            try:
                smd[prop]
            except:
                smd[prop] = init[prop]
    else:
        smd = init_smd(version)
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
        with open(MODELS_REPO + f"{version}/static.json", "r") as f:
            static = json.load(f)
    except:
        try:
            with open(MODELS_REPO + f"{version}/static_backup.json", "r") as f:
                static = json.load(f)
        except:
            static = init_static()
    init = init_static()
    for prop in init.keys():
        try:
            static[prop]
        except:
            static[prop] = init[prop]
    with open(MODELS_REPO + f"{version}/static.json", "w") as f:
        json.dump(static, f, indent=4)
    with open(MODELS_REPO + f"{version}/static_backup.json", "w") as f:
        json.dump(static, f, indent=4)
    return static


class CP(TypedDict):
    name: str
    version: str
    path: str
    iteration: int
    total_time: int
    max_acc: float
    max_val_acc: float
    max_patience: int
    min_loss: float
    loss_history: List[float]
    acc_history: List[float]
    val_acc_history: List[float]


def init_cp(name: str, version: str) -> CP:
    path = MODELS_REPO + f"{version}" + "/{}/"
    Path(path.format("reg")).mkdir(parents=True, exist_ok=True)
    Path(path.format("jit")).mkdir(parents=True, exist_ok=True)
    Path(path.format("cp")).mkdir(parents=True, exist_ok=True)
    path += name
    cp: CP = {
        "name": name,
        "version": version,
        "path": path,
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


def load_cp(name: str, version: str, fresh: bool) -> CP:
    if not fresh:
        try:
            with open(MODELS_REPO + f"{version}/cp/{name}.json", "r") as f:
                cp = json.load(f)
        except:
            try:
                with open(MODELS_REPO + f"{version}/cp/{name}_backup.json", "r") as f:
                    cp = json.load(f)
            except:
                cp = init_cp(name, version)
        init = init_cp(name, version)
        for prop in init.keys():
            try:
                cp[prop]
            except:
                cp[prop] = init[prop]
    else:
        cp = init_cp(name, version)
    return cp


def check_point(model: nn.Module, cp: CP) -> None:
    model = model.to(torch.device("cpu"))
    if cp["name"] == "meme_clf":
        torch.save(model.features, MODELS_REPO + cp["version"] + "/reg/features.pt")
        torch.save(
            model.features, MODELS_REPO + cp["version"] + "/reg/features_backup.pt",
        )
        jit.save(
            cast(ScriptModule, jit.script(model.features)),
            MODELS_REPO + cp["version"] + "/jit/features.pt",
        )
        jit.save(
            cast(ScriptModule, jit.script(model.features)),
            MODELS_REPO + cp["version"] + "/jit/features_backup.pt",
        )
        jit.save(
            cast(ScriptModule, jit.script(model.dense)),
            MODELS_REPO + cp["version"] + "/jit/dense.pt",
        )
        jit.save(
            cast(ScriptModule, jit.script(model.dense)),
            MODELS_REPO + cp["version"] + "/jit/dense_backup.pt",
        )
    torch.save(model, cp["path"].format("reg") + ".pt")
    torch.save(model, cp["path"].format("reg") + "_backup.pt")
    jit.save(cast(ScriptModule, jit.script(model)), cp["path"].format("jit") + ".pt")
    jit.save(
        cast(ScriptModule, jit.script(model)), cp["path"].format("jit") + "_backup.pt",
    )
    with open(cp["path"].format("cp") + ".json", "w") as f:
        json.dump(cp, f, indent=4)
    with open(cp["path"].format("cp") + "_backup.json", "w") as f:
        json.dump(cp, f, indent=4)
    model = model.to(torch.device("cuda:0"))


def avg_n(listy: List[float], avg: int) -> List[float]:
    return [np.mean(listy[i : i + avg]) for i in range(0, len(listy), avg)]

import random
import time
from random import shuffle
from typing import Any, Callable, Iterator, List, Tuple, Union, cast

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from decouple import config
from IPython.core.display import clear_output
from IPython.display import display
from PIL import Image as Img
from sqlalchemy.sql.elements import ClauseElement
from sqlalchemy.sql.expression import and_
from src.constants import MODELS_REPO, TRAINING_VERSION
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
from src.utils.display import display_df, display_template
from src.utils.model_func import (
    CP,
    Static,
    avg_n,
    check_point,
    get_static_names,
    load_cp,
)
from torch import Tensor, cuda
from torch.nn import BCELoss
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms

s3 = boto3.resource(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY")
)
bucket = s3.Bucket("memehub")

device = torch.device("cuda:0" if cuda.is_available() else "cpu")


plot_limit = 1000
transformations: Callable[..., torch.Tensor] = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Stonk(nn.Module):
    def __init__(self):
        super(Stonk, self).__init__()
        self.features_to_stonk = nn.Sequential(
            nn.Flatten(), nn.Linear(25088, 100), nn.Linear(100, 1), nn.Sigmoid(),
        )
        self.stonk_opt = SGD(
            self.features_to_stonk.parameters(),
            lr=0.000_000_1,
            momentum=0.9,
            dampening=0,
            weight_decay=0,
            nesterov=True,
        )
        self.loss_func: Callable[..., torch.Tensor] = BCELoss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.features_to_stonk(input))

    def train_step(self, batch: torch.Tensor, labels: torch.Tensor) -> float:
        self.stonk_opt.zero_grad()
        pred = self.forward(batch)
        stonk_loss = self.loss_func(pred, labels)
        stonk_loss.backward(torch.ones_like(stonk_loss))
        _ = self.stonk_opt.step()
        loss = cast(float, stonk_loss.mean().detach().cpu().item())
        return loss


class StonkTrainer:
    def __init__(self, name: str, fresh: bool, version: str = TRAINING_VERSION):
        self.fresh = fresh
        self.cp = load_cp(name, version, fresh)
        self.static = get_static_names(version)
        self.patience = 0
        try:
            self.features: nn.Module = torch.load(
                MODELS_REPO + f"{version}/reg/features.pt"
            ).to(device)
        except:
            self.features: nn.Module = torch.load(
                MODELS_REPO + f"{version}/reg/features_backup.pt"
            ).to(device)
        self.features = self.features.eval()
        self.model = self.get_model().to(device)

    def get_num_correct(self, is_validation: bool) -> Tuple[int, int]:
        with torch.no_grad():
            correct = 0
            total = 0
            for (inputs, labels) in cast(
                Iterator[Tuple[torch.Tensor, torch.Tensor]],
                DataLoader(
                    StonkSet(self.cp, self.static, is_validation=is_validation),
                    batch_size=64,
                    num_workers=1,
                    collate_fn=cast(Any, None),
                ),
            ):
                pred = self.forward(inputs.to(device))
                sum_tensor = cast(torch.Tensor, sum(pred == labels.to(device)))
                correct += int(sum_tensor.cpu().detach().item())
                total += len(labels)
            return correct, total

    def train(self, num_epochs: int = 1000) -> Iterator[CP]:
        self.num_epochs = num_epochs
        self.begin = time.time()
        self.now = time.time()
        self.losses = []
        for self.epoch in range(1, num_epochs):
            for (inputs, labels) in cast(
                Iterator[Tuple[torch.Tensor, torch.Tensor]],
                DataLoader(
                    StonkSet(self.cp, self.static, is_validation=False),
                    batch_size=64,
                    num_workers=64,
                    collate_fn=cast(Any, None),
                ),
            ):
                if labels.size()[0] > 1:
                    with torch.no_grad():
                        inputs = self.features(inputs.to(device))
                    self.losses.append(
                        self.model.train_step(
                            inputs, labels.to(device, dtype=torch.float)
                        )
                    )
            self.update_cp()
            check_point(self.model, self.cp)
            clear_output()
            self.print_stats()
            yield self.cp

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.round(self.model(self.features(images)))

    def test_model(self, only_wrong: bool = False) -> Union[str, None]:
        self.model = self.model.eval()
        for (image, num) in cast(
            Iterator[Tuple[torch.Tensor, int]],
            StonkSet(self.cp, self.static, is_validation=False),
        ):
            clear_output()
            name = self.humanize_pred(num)
            pred = self.predict(image)
            if only_wrong and name == pred:
                continue
            print(f"Model - {self.cp['name']}")
            print(f"Target - {name}")
            print(f"Result - {pred}")
            image = transforms.ToPILImage()(image).convert("RGB")
            print("Meme/Template")
            _ = display(image)
            display_template(self.cp["name"])
            if (
                (
                    keypress := input(
                        """
                        hit enter for next image
                        q to break
                        c to continue
                        k to kill
                        """
                    )
                )
                in "qck"
                and keypress
            ):
                return keypress

    def predict(self, image: torch.Tensor) -> bool:
        with torch.no_grad():
            pred = cast(
                float,
                self.forward(torch.tensor([image.numpy()]).to(device))
                .cpu()
                .detach()
                .item(),
            )
            return self.humanize_pred(pred)

    def get_model(self) -> Stonk:
        if self.fresh:
            model = Stonk()
        else:
            try:
                model: Stonk = torch.load(self.cp["path"].format("reg") + ".pt")
            except:
                try:
                    model: Stonk = torch.load(
                        self.cp["path"].format("reg") + "_backup.pt"
                    )
                except:
                    model = Stonk()
        return model

    @staticmethod
    def humanize_pred(pred: float) -> bool:
        return bool(pred)

    def print_stats(self) -> None:
        clear_output()
        self.display_cp()

    def update_cp(self) -> None:
        self.model_runtime = int(time.time() - self.now)
        self.correct, self.total = self.get_num_correct(is_validation=False)
        new_acc = self.correct / self.total
        new_loss: float = np.mean(self.losses)
        self.losses: List[float] = []
        self.cp["min_loss"] = min(new_loss, self.cp["min_loss"])
        self.cp["loss_history"].append(new_loss)
        if new_acc > self.cp["max_acc"]:
            self.cp["max_acc"] = new_acc
            self.patience = 0
        else:
            self.patience += 1
            self.cp["max_patience"] = max(self.patience, self.cp["max_patience"])
        self.cp["acc_history"].append(new_acc)
        val_correct, val_total = self.get_num_correct(is_validation=True)
        new_val_acc = val_correct / val_total
        self.cp["val_acc_history"].append(new_val_acc)
        if new_val_acc > self.cp["max_val_acc"]:
            self.cp["max_val_acc"] = new_val_acc
        self.cp["iteration"] += 1
        self.uptime = int(cast(float, np.round(time.time() - self.begin)))
        self.cp["total_time"] += np.round(time.time() - self.now)
        self.now = time.time()

    def display_cp(self) -> None:
        display_df(
            pd.DataFrame.from_records(
                [
                    dict(
                        name=self.cp["name"],
                        iteration=self.cp["iteration"],
                        num_correct=f"{self.correct}/{self.total}",
                        current_acc=self.cp["acc_history"][-1],
                        current_val_acc=self.cp["val_acc_history"][-1],
                        patience=self.patience,
                        max_acc=self.cp["max_acc"],
                        max_val_acc=self.cp["max_val_acc"],
                        max_patience=self.cp["max_patience"],
                        min_loss=self.cp["min_loss"],
                        num_left=self.num_epochs - self.epoch,
                        version=self.cp["version"],
                    )
                ]
            )
        )

    def summary(self):
        self.print_stats()
        self.print_graphs()

    def print_graphs(self) -> None:
        avg = len(self.cp["acc_history"]) // 100 + 1
        avg_loss_history = avg_n(self.cp["loss_history"], avg)
        avg_acc_history = avg_n(self.cp["acc_history"], avg)
        avg_val_acc_history = avg_n(self.cp["val_acc_history"], avg)
        # plt.figure(figsize=(14, 5))
        # plt.ticklabel_format(style="plain", useOffset=False)
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
        fig.tight_layout()
        ax[0, 0].plot(
            range(len(avg_loss_history[-plot_limit:])), avg_loss_history[-plot_limit:],
        )
        ax[1, 0].plot(
            range(len(avg_acc_history[-plot_limit:])), avg_acc_history[-plot_limit:]
        )
        ax[1, 1].plot(
            range(len(avg_val_acc_history[-plot_limit:])),
            avg_val_acc_history[-plot_limit:],
        )
        ax[0, 0].grid()
        ax[1, 0].grid()
        ax[1, 1].grid()
        ax[0, 0].set_title("Loss")
        ax[1, 0].set_title("Accuracy")
        ax[1, 1].set_title("Validation Accuracy")
        plt.show()


class StonkSet(IterableDataset[Dataset[torch.Tensor]]):
    def __init__(self, cp: CP, static: Static, is_validation: bool):
        self.name = cp["name"]
        max_name_idx = static["max_name_idx"]
        self.names = static["names_to_shuffle"].copy()
        if is_validation:
            self.correct_entity = MemeCorrectTest
            self.incorrect_entity = MemeIncorrectTest
            self.not_a_meme_entity = NotAMemeTest
            self.not_a_template_entity = NotATemplateTest
            if self.name == "not_a_meme":
                self.my_max_correct = max_name_idx["test"]["not_a_meme"]
                self.my_max_incorrect = 0
            elif self.name == "not_a_template":
                self.my_max_correct = max_name_idx["test"]["not_a_template"]
                self.my_max_incorrect = 0
            else:
                self.my_max_correct = max_name_idx["test"]["correct"][self.name]
                self.my_max_incorrect = max_name_idx["test"]["incorrect"][self.name]
            self.max_name_idx = max_name_idx["test"]
        else:
            self.correct_entity = MemeCorrectTrain
            self.incorrect_entity = MemeIncorrectTrain
            self.not_a_meme_entity = NotAMemeTrain
            self.not_a_template_entity = NotATemplateTrain
            if self.name == "not_a_meme":
                self.my_max_correct = max_name_idx["train"]["not_a_meme"]
                self.my_max_incorrect = 0
            elif self.name == "not_a_template":
                self.my_max_correct = max_name_idx["train"]["not_a_template"]
                self.my_max_incorrect = 0
            else:
                self.my_max_correct = max_name_idx["train"]["correct"][self.name]
                self.my_max_incorrect = max_name_idx["train"]["incorrect"][self.name]
            self.max_name_idx = max_name_idx["train"]
        self.cpb = 1 / 4
        if self.my_max_incorrect:
            self.ipb = min(1 / 4, max(1 / 100, self.my_max_incorrect / len(self.names)))
        else:
            self.ipb = 0

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        shuffle(self.names)
        while len(self.names) != 0:
            rand = random.random()
            if rand < self.cpb:
                yield self.get_correct()
            elif 1 - self.ipb < rand:
                yield self.get_incorrect()
            else:
                yield self.get_assortment()

    def get_incorrect(self):
        return (
            transformations(
                Img.open(
                    cast(
                        Tuple[str],
                        training_db.query(self.incorrect_entity.path)
                        .filter(
                            and_(
                                cast(
                                    ClauseElement,
                                    self.incorrect_entity.name == self.name,
                                ),
                                cast(
                                    ClauseElement,
                                    self.correct_entity.name_idx
                                    == random.randint(0, self.my_max_incorrect),
                                ),
                            )
                        )
                        .first(),
                    )[0]
                )
            ),
            0,
        )

    def get_correct(self) -> Tuple[Tensor, int]:
        return (
            transformations(
                Img.open(
                    cast(
                        Tuple[str],
                        training_db.query(self.correct_entity.path)
                        .filter(
                            and_(
                                cast(
                                    ClauseElement,
                                    self.correct_entity.name == self.name,
                                ),
                                cast(
                                    ClauseElement,
                                    self.correct_entity.name_idx
                                    == random.randint(0, self.my_max_correct),
                                ),
                            )
                        )
                        .first(),
                    )[0]
                )
            ),
            1,
        )

    def get_assortment(self):
        image_name = self.names.pop()
        if image_name == "not_a_meme":
            entity = self.not_a_meme_entity
            rand = random.randint(0, self.max_name_idx["not_a_meme"])
            clause = cast(ClauseElement, entity.name_idx == rand)
        elif image_name == "not_a_tempalte":
            entity = self.not_a_template_entity
            rand = random.randint(0, self.max_name_idx["not_a_template"])
            clause = cast(ClauseElement, entity.name_idx == rand)
        else:
            entity = self.correct_entity
            rand = random.randint(0, self.max_name_idx["correct"][image_name])
            clause = and_(
                cast(ClauseElement, entity.name == image_name),
                cast(ClauseElement, entity.name_idx == rand),
            )
        p = cast(Tuple[str], training_db.query(entity.path).filter(clause).first())[0]
        return (transformations(Img.open(p)), 1 if image_name == self.name else 0)


import json
import random
import time
from typing import Any, Callable, Dict, Iterator, List, Tuple, Union, cast

import arrow
import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.cuda as cuda
import torch.nn as nn
from decouple import config
from IPython.core.display import clear_output
from IPython.display import display
from pandas.core.series import Series
from PIL import Image as Img
from sqlalchemy.sql.elements import ClauseElement
from sqlalchemy.sql.expression import and_
from src.constants import TRAINING_VERSION
from src.schema import (
    MemeCorrectTest,
    MemeCorrectTrain,
    NotAMemeTest,
    NotAMemeTrain,
    NotATemplateTest,
    NotATemplateTrain,
)
from src.session import training_db
from src.utils.data_folders import name_to_img_count
from src.utils.display import display_df
from src.utils.model_func import (
    TestTrainToMax,
    avg_n,
    check_point,
    get_static_names,
    load_cp,
)
from src.utils.secondToText import secondsToText
from torch import cuda
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms
from torchvision.models import vgg16

transformations: Callable[..., torch.Tensor] = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
s3 = boto3.resource(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY")
)
bucket = s3.Bucket("memehub")
device = torch.device("cuda:0" if cuda.is_available() else "cpu")


class MemeClf(nn.Module):
    def __init__(self, output_size: int):
        super(MemeClf, self).__init__()
        self.features = nn.Sequential(*(list(vgg16(pretrained=True).children())[:-1]))
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, output_size),
            nn.Softmax(dim=1),
        )
        self.features_opt = SGD(
            self.features.parameters(),
            lr=0.001,
            momentum=0.9,
            dampening=0,
            weight_decay=0,
            nesterov=True,
        )
        self.dense_opt = SGD(
            self.dense.parameters(),
            lr=0.001,
            momentum=0.9,
            dampening=0,
            weight_decay=0,
            nesterov=True,
        )
        self.loss: Callable[..., torch.Tensor] = nn.CrossEntropyLoss()

    def train_step(self, batch: torch.Tensor, labels: torch.Tensor) -> float:
        self.features_opt.zero_grad()
        self.dense_opt.zero_grad()
        loss = self.loss(self.dense(self.features(batch)), labels)
        loss.backward(torch.ones_like(loss))
        _ = self.features_opt.step()
        _ = self.dense_opt.step()
        return cast(float, loss.detach().cpu().item())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.dense(self.features(images)), dim=1)


class MemeClfSet(IterableDataset[Dataset[torch.Tensor]]):
    def __init__(
        self,
        max_name_idx: TestTrainToMax,
        names_to_shuffle: List[str],
        name_num: Dict[str, int],
        is_validation: bool,
    ):
        self.name_num = name_num
        self.names_to_shuffle = names_to_shuffle
        if is_validation:
            self.meme_entity = MemeCorrectTest
            self.not_a_meme_entity = NotAMemeTest
            self.not_a_template_entity = NotATemplateTest
            self.max_name_idx = max_name_idx["test"]
        else:
            self.meme_entity = MemeCorrectTrain
            self.not_a_meme_entity = NotAMemeTrain
            self.not_a_template_entity = NotATemplateTrain
            self.max_name_idx = max_name_idx["train"]

    def __iter__(self):
        random.shuffle(self.names_to_shuffle)
        for name in self.names_to_shuffle:
            if name == "not_a_meme":
                entity = self.not_a_meme_entity
                rand = random.randint(0, self.max_name_idx["not_a_meme"])
                clause = cast(ClauseElement, entity.name_idx == rand)
            elif name == "not_a_template":
                entity = self.not_a_template_entity
                rand = random.randint(0, self.max_name_idx["not_a_template"])
                clause = cast(ClauseElement, entity.name_idx == rand)
            else:
                entity = self.meme_entity
                rand = random.randint(0, self.max_name_idx["correct"][name])
                clause = and_(
                    cast(ClauseElement, entity.name == name),
                    cast(ClauseElement, entity.name_idx == rand),
                )
            path = cast(
                Tuple[str], training_db.query(entity.path).filter(clause).first()
            )[0]
            yield (
                transformations(Img.open(path)),
                self.name_num[name],
            )


class MemeClfTrainer:
    def __init__(self, version: str = TRAINING_VERSION) -> None:
        self.patience = 0
        if input("Do you want fresh?") == "y":
            self.fresh = True
        else:
            self.fresh = False
        self.cp = load_cp("meme_clf", version, self.fresh)
        self.static = get_static_names(version)
        self.model = self.get_model().to(device)

    def train_epoch(self, dataset: MemeClfSet, batch_size: int, num_workers: int):
        for (inputs, labels) in cast(
            Iterator[Tuple[torch.Tensor, torch.Tensor]],
            DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=cast(Any, None),
            ),
        ):
            self.losses.append(
                self.model.train_step(inputs.to(device), labels.to(device))
            )

    def train(
        self,
        trigger1: float,
        trigger2: float,
        finish: float,
        num_workers: int,
        num_epochs: int,
    ) -> None:
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.trigger1 = trigger1
        self.trigger2 = trigger2
        self.finish = finish
        self.losses: List[float] = []
        self.begin = time.time()
        self.now = time.time()
        if self.cp["max_acc"] <= self.trigger1:
            self.train_full()
        self.train_wrong_names()

    def train_full(self):
        for self.epoch in range(1, self.num_epochs):
            self.train_epoch(
                MemeClfSet(
                    self.static["max_name_idx"],
                    self.static["names_to_shuffle"],
                    self.static["name_num"],
                    is_validation=False,
                ),
                64,
                self.num_workers,
            )
            self.update_cp()
            check_point(self.model, self.cp)
            clear_output()
            self.print_stats()
            if self.cp["max_acc"] > self.trigger1:
                break

    def train_wrong_names(self):
        self.now = time.time()
        self.losses: List[float] = []
        wrong_names = self.get_wrong_names()
        for self.epoch in range(1, self.num_epochs):
            rounds = 10 if self.cp["max_acc"] < self.trigger2 else 3
            for _ in range(rounds):
                self.train_epoch(
                    MemeClfSet(
                        self.static["max_name_idx"],
                        wrong_names,
                        self.static["name_num"],
                        is_validation=False,
                    ),
                    batch_size=64,
                    num_workers=self.num_workers,
                )
                self.update_cp()
                clear_output()
                print(f"num wrong names - {len(wrong_names)}")
                self.print_stats()
                print(wrong_names)
                if self.cp["max_acc"] > self.finish:
                    break
            wrong_names = self.get_wrong_names()

    def get_num_correct(self, is_validation: bool) -> Tuple[int, int]:
        with torch.no_grad():
            correct = 0
            total = 0
            for (inputs, labels) in cast(
                Iterator[Tuple[torch.Tensor, torch.Tensor]],
                DataLoader(
                    MemeClfSet(
                        self.static["max_name_idx"],
                        self.static["names_to_shuffle"],
                        self.static["name_num"],
                        is_validation=is_validation,
                    ),
                    batch_size=64,
                    num_workers=0,
                    collate_fn=cast(Any, None),
                ),
            ):
                pred = self.model(inputs.to(device))
                sum_tensor = cast(torch.Tensor, sum(pred == labels.to(device)))
                correct += int(sum_tensor.cpu().detach().item())
                total += len(labels)
            return correct, total

    def get_wrong_names(self) -> List[str]:
        with torch.no_grad():
            wrong_names = {name: 0 for name in self.static["names"]}
            for (inputs, labels) in cast(
                Iterator[Tuple[torch.Tensor, torch.Tensor]],
                DataLoader(
                    MemeClfSet(
                        self.static["max_name_idx"],
                        self.static["names_to_shuffle"],
                        self.static["name_num"],
                        is_validation=False,
                    ),
                    batch_size=64,
                    num_workers=64,
                    collate_fn=cast(Any, None),
                ),
            ):
                preds = self.model(inputs.to(device)).cpu().detach().numpy()
                labels = labels.numpy()
                for label, pred in cast(Iterator[Tuple[int, int]], zip(labels, preds)):
                    if label != pred:
                        wrong_names[self.humanize_pred(label)] += 1
            return list(
                map(
                    lambda x: x[0],
                    filter(lambda x: x[1] > 0, list(wrong_names.items())),
                )
            )

    def test_model(self, only_wrong: bool = False):
        self.model = self.model.eval()
        self.manual = True
        wrong_names = {name: 0 for name in self.static["names"]}
        for (image, num) in MemeClfSet(
            self.static["max_name_idx"],
            self.static["names_to_shuffle"],
            self.static["name_num"],
            is_validation=False,
        ):
            clear_output()
            name = self.humanize_pred(num)
            pred = self.predict(image)
            if pred != name:
                wrong_names[pred] += 1
            if self.manual:
                if only_wrong and name == pred:
                    continue
                print(f"Model - {self.cp['name']}")
                print(f"Target - {name}")
                print(f"Result - {pred}")
                image = transforms.ToPILImage()(image).convert("RGB")
                print("Meme/Template")
                _ = display(image)
                self.display_template(name)
            if (
                self.manual
                and (
                    keypress := input(
                        """
                        hit enter for next image
                        q to break
                        c to continue
                        k to kill
                        """
                    )
                )
                in "qckm"
                and keypress
            ):
                if keypress == "m":
                    self.manual = False
                else:
                    return keypress

    def predict(self, image: torch.Tensor) -> str:
        with torch.no_grad():
            image = torch.tensor([image.numpy()])
            pred = cast(int, self.model(image.to(device)).cpu().detach().item())
            return self.humanize_pred(pred)

    def get_model(self) -> MemeClf:
        if self.fresh:
            model = MemeClf(output_size=len(self.static["names"]))
        else:
            try:
                model: MemeClf = torch.load(self.cp["path"].format("reg") + ".pt")
            except:
                try:
                    model: MemeClf = torch.load(
                        self.cp["path"].format("reg") + "_backup.pt"
                    )
                except:
                    model = MemeClf(output_size=len(self.static["names"]))
        return model

    def humanize_pred(self, pred: int) -> str:
        return self.static["num_name"][str(pred)]

    def print_stats(self) -> None:
        self.display_cp()
        num_left = self.num_epochs - self.epoch
        eta = (self.cp["total_time"] * num_left) // self.cp["iteration"]
        print(
            json.dumps(
                dict(
                    timestamp=cast(str, arrow.utcnow().to("local").format("HH:mm:ss")),
                    model_runtime=secondsToText(self.model_runtime),
                    uptime=secondsToText(self.uptime),
                    total_time=secondsToText(self.cp["total_time"]),
                    eta=secondsToText(eta),
                ),
                indent=0,
            )[1:-1].replace('"', "")
        )
        self.print_graphs()

    def update_cp(self) -> None:
        self.model_runtime = int(time.time() - self.now)
        self.correct, self.total = self.get_num_correct(is_validation=False)
        new_acc = self.correct / self.total
        new_loss: float = np.mean(self.losses)
        self.losses: List[float] = []
        self.cp["min_loss"] = min(new_loss, self.cp["min_loss"])
        self.cp["loss_history"].append(new_loss)
        self.cp["acc_history"].append(new_acc)
        val_correct, val_total = self.get_num_correct(is_validation=True)
        new_val_acc = val_correct / val_total
        self.cp["val_acc_history"].append(new_val_acc)
        if new_val_acc > self.cp["max_val_acc"]:
            self.cp["max_val_acc"] = new_val_acc
        self.cp["iteration"] += 1
        self.uptime = int(cast(float, np.round(time.time() - self.begin)))
        self.cp["total_time"] += np.round(time.time() - self.now)
        if new_acc > self.cp["max_acc"]:
            self.cp["max_acc"] = new_acc
            self.patience = 0
            check_point(self.model, self.cp)
        else:
            self.patience += 1
            self.cp["max_patience"] = max(self.patience, self.cp["max_patience"])
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

    def display_wrong_names(self) -> None:
        df = pd.DataFrame(
            list(self.get_wrong_names().items()), columns=["name", "num_wrong"]
        )
        df = df[df["num_wrong"] != 0]
        df["img_count"] = cast(Series[str], df["name"]).apply(name_to_img_count)
        display_df(df.sort_values("num_wrong", ascending=False))

    def display_template(self, name: Union[str, bool]):
        try:
            object = bucket.Object(
                "memehub/templates/" + self.cp["name"]
                if isinstance(name, bool)
                else name
            )
            response = object.get()
            file_stream = response["Body"]
            template = Img.open(file_stream)
            _ = display(template)
        except:
            pass

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
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 6))
        fig.tight_layout()
        ax[0, 0].plot(range(len(avg_loss_history)), avg_loss_history)
        ax[1, 0].plot(range(len(avg_acc_history)), avg_acc_history)
        ax[1, 1].plot(range(len(avg_val_acc_history)), avg_val_acc_history)
        ax[0, 0].grid()
        ax[1, 0].grid()
        ax[1, 1].grid()
        ax[0, 0].set_title("Loss Full")
        ax[1, 0].set_title("Accuracy Full")
        ax[1, 1].set_title("Validation Accuracy Full")
        ax[0, 2].plot(range(100), self.cp["loss_history"][-100:])
        ax[1, 2].plot(range(100), self.cp["acc_history"][-100:])
        ax[1, 3].plot(range(100), self.cp["val_acc_history"][-100:])
        ax[0, 2].grid()
        ax[1, 2].grid()
        ax[1, 3].grid()
        ax[0, 2].set_title("Loss End")
        ax[1, 2].set_title("Accuracy End")
        ax[1, 3].set_title("Validation Accuracy End")
        plt.show()

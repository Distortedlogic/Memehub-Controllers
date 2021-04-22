import json
import random
import time
from itertools import chain, repeat
from random import shuffle
from typing import Any, Callable, Iterator, List, Tuple, Union, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython.core.display import clear_output
from IPython.display import display
from PIL import Image as Img
from sqlalchemy.sql.elements import ClauseElement
from sqlalchemy.sql.expression import and_
from src.constants import MEME_CLF_REPO, STONK_REPO, backup
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
from src.trainers.trainer import Trainer
from src.utils.display import display_template
from src.utils.model_func import CP, TestTrainToMax, avg_n, load_cp
from src.utils.transforms import toTensorOnly, trainingTransforms
from torch import Tensor, cuda, jit
from torch._C import ScriptModule
from torch.nn import BCELoss
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms

device = torch.device("cuda:0" if cuda.is_available() else "cpu")


class Stonk(nn.Module):
    def __init__(self, size: int):
        super(Stonk, self).__init__()
        self.features_to_stonk = nn.Sequential(
            nn.Flatten(), nn.Linear(25088, size), nn.Linear(size, 1), nn.Sigmoid(),
        )
        self.stonk_opt = SGD(
            self.features_to_stonk.parameters(),
            lr=0.000_01,
            momentum=0.9,
            dampening=0,
            weight_decay=0,  # 0.000_001,
            nesterov=True,
        )
        self.loss_func: Callable[..., torch.Tensor] = BCELoss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.features_to_stonk(input))

    def train_step(self, batch: torch.Tensor, labels: torch.Tensor) -> float:
        self.stonk_opt.zero_grad()
        stonk_loss = self.loss_func(self.forward(batch), labels)
        stonk_loss.backward(torch.ones_like(stonk_loss))
        _ = self.stonk_opt.step()
        loss = cast(float, stonk_loss.mean().detach().cpu().item())
        return loss


class StonkTrainer(Trainer):
    def __init__(self, name: str, fresh: bool):
        self.fresh = fresh
        super(StonkTrainer, self).__init__()
        self.patience = 0
        self.cp = load_cp(STONK_REPO.format("cp") + name, self.fresh)
        self.name = name
        try:
            self.features: nn.Module = torch.load(
                MEME_CLF_REPO.format("reg") + "features.pt"
            ).to(device)
        except Exception:
            self.features: nn.Module = torch.load(
                backup(MEME_CLF_REPO.format("reg")) + "features.pt"
            ).to(device)
        self.features = self.features.eval()
        self.model: Stonk = self.get_model(100).to(device)

    def get_num_correct(
        self, is_validation: bool, use_transforms: bool
    ) -> Tuple[int, int]:
        with torch.no_grad():
            correct = 0
            total = 0
            for (inputs, labels) in cast(
                Iterator[Tuple[torch.Tensor, torch.Tensor]],
                DataLoader(
                    StonkSet(
                        self.name,
                        self.static["names_to_shuffle"],
                        self.static["max_name_idx"],
                        is_validation=is_validation,
                        is_training=False,
                        size=1,
                        use_transforms=use_transforms,
                    ),
                    batch_size=64,
                    num_workers=16,
                    collate_fn=cast(Any, None),
                ),
            ):
                pred = self.forward(inputs.to(device))
                sum_tensor = cast(torch.Tensor, sum(pred == labels.to(device)))
                correct += int(sum_tensor.cpu().detach().item())
                total += len(labels)
            return correct, total

    def train(
        self, num_workers: int, batch_size: int, num_epochs: int = 1000
    ) -> Iterator[CP]:
        self.num_epochs = num_epochs
        self.begin = int(time.time())
        self.now = time.time()
        self.losses = []
        for self.epoch in range(1, num_epochs):
            for (inputs, labels) in cast(
                Iterator[Tuple[torch.Tensor, torch.Tensor]],
                DataLoader(
                    StonkSet(
                        self.name,
                        self.static["names_to_shuffle"],
                        self.static["max_name_idx"],
                        is_validation=False,
                        size=4,
                        use_transforms=True,
                        is_training=True,
                    ),
                    batch_size=batch_size,
                    num_workers=num_workers,
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
            clear_output()
            self.print_stats()
            yield self.cp

    def update_loss(self):
        self.cp["min_loss"] = min(min(self.losses), self.cp["min_loss"])
        self.cp["loss_history"] += self.losses
        self.new_loss = self.losses[-1]
        self.losses: List[float] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.round(self.model(self.features(images)))

    def test_model(self, only_wrong: bool = False) -> Union[str, None]:
        self.model = self.model.eval()
        for (image, num) in cast(
            Iterator[Tuple[torch.Tensor, int]],
            StonkSet(
                self.name,
                self.static["names_to_shuffle"],
                self.static["max_name_idx"],
                is_validation=False,
                size=1,
                is_training=False,
                use_transforms=False,
            ),
        ):
            clear_output()
            name = self.humanize_pred(num)
            pred = self.predict(image)
            if only_wrong and name == pred:
                continue
            print(f"Model - {self.name}")
            print(f"Target - {name}")
            print(f"Result - {pred}")
            image = transforms.ToPILImage()(image).convert("RGB")
            print("Meme/Template")
            _ = display(image)
            display_template(self.name)
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

    def get_model(self, size: int) -> Stonk:
        if self.fresh:
            model = Stonk(size)
        else:
            try:
                model: Stonk = torch.load(STONK_REPO.format("reg") + self.name + ".pt")
            except Exception as e:
                print(e)
                try:
                    model: Stonk = torch.load(
                        backup(STONK_REPO.format("reg")) + self.name + ".pt"
                    )
                except Exception as e:
                    print(e)
                    model = Stonk(size)
        return model

    def check_point(self, is_backup: bool) -> None:
        self.model = self.model.to(torch.device("cpu"))
        REPO = backup(STONK_REPO) if is_backup else STONK_REPO
        torch.save(self.model, REPO.format("reg") + self.name + ".pt")
        jit.save(
            cast(ScriptModule, jit.script(self.model)),
            REPO.format("jit") + self.name + ".pt",
        )
        with open(REPO.format("cp") + self.name + ".json", "w") as f:
            json.dump(self.cp, f, indent=4)
        self.model = self.model.to(torch.device("cuda:0"))

    @staticmethod
    def humanize_pred(pred: float) -> bool:
        return bool(pred)

    def print_stats(self) -> None:
        clear_output()
        self.display_cp()
        self.print_graphs()

    def print_graphs(self) -> None:
        _ = plt.figure(figsize=(20, 6))
        # plt.ticklabel_format(style="plain", useOffset=False)
        _ = plt.plot(range(len(self.cp["loss_history"])), self.cp["loss_history"])
        plt.grid()
        _ = plt.title("Loss Full")
        plt.show()


class StonkSet(IterableDataset[Dataset[torch.Tensor]]):
    def __init__(
        self,
        name: str,
        names: List[str],
        max_name_idx: TestTrainToMax,
        is_validation: bool,
        size: int,
        is_training: bool,
        use_transforms: bool,
        cpb: float = 2 / 5,
        nmpb: float = 1 / 10,
        ntpb: float = 1 / 5,
        ipb: float = 1 / 10,
    ):
        self.name = name
        self.names = names
        self.is_training = is_training
        self.use_transforms = use_transforms
        if is_training:
            self.names = list(chain.from_iterable(repeat(names, size)))
        if use_transforms:
            self.transforms = trainingTransforms
        else:
            self.transforms = toTensorOnly
        if is_validation:
            self.correct_entity = MemeCorrectTest
            self.incorrect_entity = MemeIncorrectTest
            self.not_a_meme_entity = NotAMemeTest
            self.not_a_template_entity = NotATemplateTest
            self.my_max_correct = max_name_idx["test"]["correct"][self.name]
            self.my_max_incorrect = max_name_idx["test"]["incorrect"][self.name]
            self.max_name_idx = max_name_idx["test"]
        else:
            self.correct_entity = MemeCorrectTrain
            self.incorrect_entity = MemeIncorrectTrain
            self.not_a_meme_entity = NotAMemeTrain
            self.not_a_template_entity = NotATemplateTrain
            self.my_max_correct = max_name_idx["train"]["correct"][self.name]
            self.my_max_incorrect = max_name_idx["train"]["incorrect"][self.name]
            self.max_name_idx = max_name_idx["train"]
        self.cpb = cpb
        self.nmpb = nmpb
        max_not_a_template = self.max_name_idx["not_a_template"]
        if max_not_a_template:
            self.ntpb = min(ntpb, max(1 / 100, max_not_a_template / len(self.names)),)
        else:
            self.ntpb = 0
        if self.my_max_incorrect:
            self.ipb = min(ipb, max(1 / 100, self.my_max_incorrect / len(self.names)))
        else:
            self.ipb = 0

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        shuffle(self.names)
        while len(self.names) != 0:
            rand = random.random()
            if rand < self.cpb:
                yield self.get_correct()
            elif self.cpb < rand < self.nmpb:
                yield self.get_not_meme()
            elif 1 - self.ipb < rand:
                yield self.get_incorrect()
            elif 1 - self.ipb - self.ntpb < rand < 1 - self.ipb:
                yield self.get_not_template()
            else:
                yield self.get_assortment()

    def get_incorrect(self):
        return (
            toTensorOnly(
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
                                    self.incorrect_entity.name_idx
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

    def get_not_template(self):
        return (
            toTensorOnly(
                Img.open(
                    cast(
                        Tuple[str],
                        training_db.query(self.not_a_template_entity.path)
                        .filter(
                            cast(
                                ClauseElement,
                                self.not_a_template_entity.name_idx
                                == random.randint(
                                    0, self.max_name_idx["not_a_template"]
                                ),
                            ),
                        )
                        .first(),
                    )[0]
                )
            ),
            0,
        )

    def get_not_meme(self):
        return (
            toTensorOnly(
                Img.open(
                    cast(
                        Tuple[str],
                        training_db.query(self.not_a_meme_entity.path)
                        .filter(
                            cast(
                                ClauseElement,
                                self.not_a_meme_entity.name_idx
                                == random.randint(0, self.max_name_idx["not_a_meme"]),
                            ),
                        )
                        .first(),
                    )[0]
                )
            ),
            0,
        )

    def get_correct(self) -> Tuple[Tensor, int]:
        entity = self.correct_entity
        rand = random.randint(0, self.my_max_correct)
        clause = and_(
            cast(ClauseElement, entity.name == self.name),
            cast(ClauseElement, entity.name_idx == rand),
        )
        if self.is_training and self.use_transforms == None:
            transforms = toTensorOnly if random.random() < 0.15 else trainingTransforms
        else:
            if self.use_transforms:
                transforms = trainingTransforms
            else:
                transforms = toTensorOnly
        return (
            transforms(
                Img.open(
                    cast(
                        Tuple[str],
                        training_db.query(entity.path).filter(clause).first(),
                    )[0]
                )
            ),
            1,
        )

    def get_assortment(self):
        image_name = self.names.pop()
        entity = self.correct_entity
        rand = random.randint(0, self.max_name_idx["correct"][image_name])
        clause = and_(
            cast(ClauseElement, entity.name == image_name),
            cast(ClauseElement, entity.name_idx == rand),
        )
        p = cast(Tuple[str], training_db.query(entity.path).filter(clause).first())[0]
        transforms = toTensorOnly if random.random() < 0.5 else self.transforms
        return (transforms(Img.open(p)), 1 if image_name == self.name else 0)

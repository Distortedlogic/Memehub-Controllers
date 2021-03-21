import random
import time
from random import shuffle
from typing import Any, Callable, Iterator, Tuple, Union, cast

import torch
import torch.nn as nn
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
)
from src.session import training_db
from src.trainers.trainer import Trainer
from src.utils.display import display_template
from src.utils.model_func import CP, Static
from torch import Tensor, cuda
from torch.nn import BCELoss
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms

device = torch.device("cuda:0" if cuda.is_available() else "cpu")
transformations: Callable[..., torch.Tensor] = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Stonk(nn.Module):
    def __init__(self, size: int):
        super(Stonk, self).__init__()
        self.features_to_stonk = nn.Sequential(
            nn.Flatten(), nn.Linear(25088, size), nn.Linear(size, 1), nn.Sigmoid(),
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
        stonk_loss = self.loss_func(self.forward(batch), labels)
        stonk_loss.backward(torch.ones_like(stonk_loss))
        _ = self.stonk_opt.step()
        loss = cast(float, stonk_loss.mean().detach().cpu().item())
        return loss


class StonkTrainer(Trainer):
    def __init__(self, name: str, fresh: bool, version: str = TRAINING_VERSION):
        self.fresh = fresh
        super(StonkTrainer, self).__init__(name, version)
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
        size = 1000 if name in ["not_a_meme", "not_a_template"] else 100
        self.model = self.get_model(size).to(device)

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
        self.begin = int(time.time())
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

    def get_model(self, size: int) -> Stonk:
        if self.fresh:
            model = Stonk(size)
        else:
            try:
                model: Stonk = torch.load(self.cp["path"].format("reg") + ".pt")
            except:
                try:
                    model: Stonk = torch.load(
                        self.cp["path"].format("reg") + "_backup.pt"
                    )
                except:
                    model = Stonk(size)
        return model

    @staticmethod
    def humanize_pred(pred: float) -> bool:
        return bool(pred)

    def print_stats(self) -> None:
        clear_output()
        self.display_cp()


class StonkSet(IterableDataset[Dataset[torch.Tensor]]):
    def __init__(self, cp: CP, static: Static, is_validation: bool):
        self.name = cp["name"]
        max_name_idx = static["max_name_idx"]
        self.names = static["names_to_shuffle"].copy()
        if is_validation:
            self.correct_entity = MemeCorrectTest
            self.incorrect_entity = MemeIncorrectTest
            self.my_max_correct = max_name_idx["test"]["correct"][self.name]
            self.my_max_incorrect = max_name_idx["test"]["incorrect"][self.name]
            self.max_name_idx = max_name_idx["test"]
        else:
            self.correct_entity = MemeCorrectTrain
            self.incorrect_entity = MemeIncorrectTrain
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
        entity = self.correct_entity
        rand = random.randint(0, self.my_max_correct)
        clause = and_(
            cast(ClauseElement, entity.name == self.name),
            cast(ClauseElement, entity.name_idx == rand),
        )
        return (
            transformations(
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
        return (transformations(Img.open(p)), 1 if image_name == self.name else 0)


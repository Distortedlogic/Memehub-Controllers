import random
import time
from typing import Any, Callable, Iterator, List, Tuple, cast

import torch
import torch.cuda as cuda
import torch.nn as nn
from IPython.core.display import clear_output
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
from src.trainers.trainer import Trainer
from src.utils.model_func import TestTrainToMax, check_point
from torch import cuda
from torch.nn import BCELoss
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
device = torch.device("cuda:0" if cuda.is_available() else "cpu")


class NotAMemeClf(nn.Module):
    def __init__(self):
        super(NotAMemeClf, self).__init__()
        self.features = nn.Sequential(*(list(vgg16(pretrained=True).children())[:-1]))
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1),
            nn.Sigmoid(),
        )
        self.features_opt = SGD(
            self.features.parameters(),
            lr=0.01,
            momentum=0.9,
            dampening=0,
            weight_decay=0,
            nesterov=True,
        )
        self.dense_opt = SGD(
            self.dense.parameters(),
            lr=0.01,
            momentum=0.9,
            dampening=0,
            weight_decay=0,
            nesterov=True,
        )
        self.loss_func: Callable[..., torch.Tensor] = BCELoss()

    def train_step(self, batch: torch.Tensor, labels: torch.Tensor) -> float:
        self.features_opt.zero_grad()
        self.dense_opt.zero_grad()
        pred = self.forward(batch)
        loss = self.loss_func(pred, labels)
        loss.backward(torch.ones_like(loss))
        _ = self.features_opt.step()
        _ = self.dense_opt.step()
        return cast(float, loss.detach().cpu().item())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        dense = self.dense(features)
        squeezed = torch.squeeze(dense, dim=1)
        return squeezed


class NotAMemeClfSet(IterableDataset[Dataset[torch.Tensor]]):
    def __init__(
        self,
        max_name_idx: TestTrainToMax,
        names_to_shuffle: List[str],
        is_validation: bool,
    ):
        self.names = names_to_shuffle
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
        if template_count := self.max_name_idx["not_a_template"]:
            self.natpb = min(1 / 4, max(1 / 100, template_count / len(self.names)))
        else:
            self.natpb = 0
        self.mtpb = 0.25 - self.natpb

    def get_meme_template(self, name: str):
        rand = random.randint(0, self.max_name_idx["correct"][name])
        clause = and_(
            cast(ClauseElement, self.meme_entity.name == name),
            cast(ClauseElement, self.meme_entity.name_idx == rand),
        )
        path = cast(
            Tuple[str], training_db.query(self.meme_entity.path).filter(clause).first()
        )[0]
        return (
            transformations(Img.open(path)),
            0,
        )

    def get_not_a_template(self):
        rand = random.randint(0, self.max_name_idx["not_a_template"])
        clause = cast(ClauseElement, self.not_a_template_entity.name_idx == rand)
        path = cast(
            Tuple[str],
            training_db.query(self.not_a_template_entity.path).filter(clause).first(),
        )[0]
        return (
            transformations(Img.open(path)),
            0,
        )

    def get_not_a_meme(self):
        rand = random.randint(0, self.max_name_idx["not_a_meme"])
        clause = cast(ClauseElement, self.not_a_meme_entity.name_idx == rand)
        path = cast(
            Tuple[str],
            training_db.query(self.not_a_meme_entity.path).filter(clause).first(),
        )[0]
        return (
            transformations(Img.open(path)),
            1,
        )

    def __iter__(self):
        random.shuffle(self.names)
        while len(self.names) != 0:
            rand = random.random()
            if rand < 0.25 + self.mtpb:
                yield self.get_meme_template(self.names.pop())
            elif 1 - self.natpb < rand:
                yield self.get_not_a_template()
            else:
                yield self.get_not_a_meme()


class NotAMemeClfTrainer(Trainer):
    def __init__(self, version: str = TRAINING_VERSION) -> None:
        self.patience = 0
        super(NotAMemeClfTrainer, self).__init__("not_a_meme_clf", version)
        self.model: NotAMemeClf = self.get_model().to(device)

    def train_epoch(self, dataset: NotAMemeClfSet, batch_size: int, num_workers: int):
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
                self.model.train_step(
                    inputs.to(device), labels.to(device, dtype=torch.float)
                )
            )

    def train(self, num_workers: int, num_epochs: int,) -> None:
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.losses: List[float] = []
        self.begin = int(time.time())
        self.now = time.time()
        self.train_full()

    def train_full(self):
        for self.epoch in range(1, self.num_epochs):
            self.train_epoch(
                NotAMemeClfSet(
                    self.static["max_name_idx"],
                    self.static["names_to_shuffle"],
                    is_validation=False,
                ),
                64,
                self.num_workers,
            )
            self.update_cp()
            check_point(self.model, self.cp)
            clear_output()
            self.print_stats()

    def get_num_correct(self, is_validation: bool) -> Tuple[int, int]:
        with torch.no_grad():
            correct = 0
            total = 0
            for (inputs, labels) in cast(
                Iterator[Tuple[torch.Tensor, torch.Tensor]],
                DataLoader(
                    NotAMemeClfSet(
                        self.static["max_name_idx"],
                        self.static["names_to_shuffle"],
                        is_validation=is_validation,
                    ),
                    batch_size=64,
                    num_workers=1,
                    collate_fn=cast(Any, None),
                ),
            ):
                pred = self.model(inputs.to(device))
                sum_tensor = cast(torch.Tensor, sum(pred == labels.to(device)))
                correct += int(sum_tensor.cpu().detach().item())
                total += len(labels)
            return correct, total

    def predict(self, image: torch.Tensor) -> bool:
        with torch.no_grad():
            image = torch.tensor([image.numpy()])
            pred = cast(int, self.model(image.to(device)).cpu().detach().item())
            return self.humanize_pred(pred)

    def get_model(self) -> NotAMemeClf:
        if self.fresh:
            model = NotAMemeClf()
        else:
            try:
                model: NotAMemeClf = torch.load(self.cp["path"].format("reg") + ".pt")
            except:
                try:
                    model: NotAMemeClf = torch.load(
                        self.cp["path"].format("reg") + "_backup.pt"
                    )
                except:
                    model = NotAMemeClf()
        return model

    def humanize_pred(self, pred: int) -> bool:
        return bool(pred)

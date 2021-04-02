import json
import random
import time
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple, cast

import torch
import torch.cuda as cuda
import torch.nn as nn
from IPython.core.display import clear_output
from IPython.display import display
from PIL import Image as Img
from sqlalchemy.sql.elements import ClauseElement
from sqlalchemy.sql.expression import and_
from src.constants import MEME_CLF_REPO
from src.schema import MemeCorrectTest, MemeCorrectTrain
from src.session import training_db
from src.trainers.trainer import Trainer
from src.utils.display import display_template
from src.utils.model_func import TestTrainToMax, load_cp
from src.utils.transforms import toTensorOnly, trainingTransforms
from torch import cuda, jit, nn
from torch._C import ScriptModule
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms
from torchvision.models import vgg16

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
            weight_decay=0.000_001,
            nesterov=True,
        )
        self.dense_opt = SGD(
            self.dense.parameters(),
            lr=0.001,
            momentum=0.9,
            dampening=0,
            weight_decay=0.000_001,
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
        is_training: bool,
        ntpb: float = 0,
        nmpb: float = 0,
    ):
        self.name_num = name_num
        self.ntpb = ntpb
        self.nmpb = nmpb
        if is_training:
            self.transforms = trainingTransforms
            self.names = list(chain.from_iterable(repeat(names_to_shuffle, 4)))
        else:
            self.transforms = toTensorOnly
            self.names = names_to_shuffle
        if is_validation:
            self.meme_entity = MemeCorrectTest
            self.max_name_idx = max_name_idx["test"]
        else:
            self.meme_entity = MemeCorrectTrain
            self.max_name_idx = max_name_idx["train"]

    def get_assortment(self):
        name = self.names.pop()
        rand = random.randint(0, self.max_name_idx["correct"][name])
        clause = and_(
            cast(ClauseElement, self.meme_entity.name == name),
            cast(ClauseElement, self.meme_entity.name_idx == rand),
        )
        path = cast(
            Tuple[str], training_db.query(self.meme_entity.path).filter(clause).first()
        )[0]
        transforms = toTensorOnly if random.random() < 0.15 else self.transforms
        return (transforms(Img.open(path)), self.name_num[name])

    def __iter__(self):
        random.shuffle(self.names)
        while self.names:
            rand = random.random()
            if rand < self.ntpb:
                pass
            elif rand > 1 - self.nmpb:
                pass
            else:
                yield self.get_assortment()


class MemeClfTrainer(Trainer):
    def __init__(self) -> None:
        self.patience = 0
        self.name = "meme_clf"
        super(MemeClfTrainer, self).__init__()
        if input("Do you want fresh?") == "y":
            fresh = True
        else:
            fresh = False
        self.cp = load_cp(MEME_CLF_REPO.format("cp") + "meme_clf", fresh)
        self.model: MemeClf = self.get_model(len(self.static["names"]), fresh).to(
            device
        )

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
        trigger1: float = 0.8,
        trigger2: float = 0.9,
        finish: float = 0.99,
        num_workers: int = 16,
        num_epochs: int = 1000,
    ) -> None:
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.trigger1 = trigger1
        self.trigger2 = trigger2
        self.finish = finish
        self.losses: List[float] = []
        self.begin = int(time.time())
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
                    is_training=True,
                ),
                64,
                self.num_workers,
            )
            self.update_cp()
            clear_output()
            self.print_stats()
            if self.cp["max_acc"] > self.trigger1:
                break

    def hard_reset(self):
        if self.cp["max_acc"] > self.trigger1 and self.new_acc < self.trigger1 - 0.1:
            self.model: MemeClf = self.get_model(len(self.static["names"]), False).to(
                device
            )
            self.cp = load_cp(MEME_CLF_REPO.format("cp") + "meme_clf", False)
            self.wrong_names = self.get_wrong_names()
            self.num_hard_resets += 1
            return True
        else:
            return False

    def train_wrong_names(self):
        self.now = time.time()
        self.losses: List[float] = []
        self.wrong_names = self.get_wrong_names()
        for self.epoch in range(1, self.num_epochs):
            rounds = 10 if self.cp["max_acc"] < self.trigger2 else 3
            for _ in range(rounds):
                self.train_epoch(
                    MemeClfSet(
                        self.static["max_name_idx"],
                        self.wrong_names,
                        self.static["name_num"],
                        is_validation=False,
                        is_training=True,
                    ),
                    batch_size=64,
                    num_workers=self.num_workers,
                )
                self.update_cp()
                clear_output()
                print(f"num wrong names - {len(self.wrong_names)}")
                self.print_stats()
                print(self.wrong_names)
                if self.cp["max_acc"] > self.finish:
                    break
            self.wrong_names = self.get_wrong_names()

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
                        is_training=False,
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
                        is_training=False,
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
            is_training=False,
        ):
            clear_output()
            name = self.humanize_pred(num)
            pred = self.predict(image)
            if pred != name:
                wrong_names[pred] += 1
            if self.manual:
                if only_wrong and name == pred:
                    continue
                print(f"Model - {self.name}")
                print(f"Target - {name}")
                print(f"Result - {pred}")
                image = transforms.ToPILImage()(image).convert("RGB")
                print("Meme/Template")
                _ = display(image)
                display_template(name)
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

    def get_model(self, output_size: int, fresh: bool) -> MemeClf:
        if fresh:
            model = MemeClf(output_size=output_size)
        else:
            try:
                model: MemeClf = torch.load(
                    MEME_CLF_REPO.format("reg") + self.name + ".pt"
                )
            except Exception as e:
                print(e)
                print("wtf")
                try:
                    model: MemeClf = torch.load(
                        MEME_CLF_REPO.format("reg") + self.name + "_backup.pt"
                    )
                except Exception as e:
                    print(e)
                    print("wtf")
                    model = MemeClf(output_size=output_size)
        return model

    def check_point(self) -> None:
        self.model = self.model.to(torch.device("cpu"))
        torch.save(self.model.features, MEME_CLF_REPO.format("reg") + "features.pt")
        torch.save(
            self.model.features, MEME_CLF_REPO.format("reg") + "features_backup.pt",
        )
        jit.save(
            cast(ScriptModule, jit.script(self.model.features)),
            MEME_CLF_REPO.format("jit") + "features.pt",
        )
        jit.save(
            cast(ScriptModule, jit.script(self.model.features)),
            MEME_CLF_REPO.format("jit") + "features_backup.pt",
        )
        jit.save(
            cast(ScriptModule, jit.script(self.model.dense)),
            MEME_CLF_REPO.format("jit") + "dense.pt",
        )
        jit.save(
            cast(ScriptModule, jit.script(self.model.dense)),
            MEME_CLF_REPO.format("jit") + "dense_backup.pt",
        )
        torch.save(self.model, MEME_CLF_REPO.format("reg") + self.name + ".pt")
        torch.save(
            self.model, MEME_CLF_REPO.format("reg") + self.name + "_backup.pt",
        )
        with open(MEME_CLF_REPO.format("cp") + self.name + ".json", "w") as f:
            json.dump(self.cp, f, indent=4)
        with open(MEME_CLF_REPO.format("cp") + self.name + "_backup.json", "w",) as f:
            json.dump(self.cp, f, indent=4)
        self.model = self.model.to(torch.device("cuda:0"))

    def humanize_pred(self, pred: int) -> str:
        return self.static["num_name"][str(pred)]

from time import time
from typing import cast

import arrow
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display import clear_output
from src.constants import MEME_CLF_VERSION
from src.trainers.stonk import StonkTrainer
from src.utils.display import display_df, pretty_print_dict
from src.utils.model_func import dump_smd, get_smd, get_static_names
from src.utils.secondToText import secondsToText


class MarketTrainer:
    def __init__(self):
        if input("Want a fresh market?"):
            fresh = True
        else:
            fresh = False
        self.smd = get_smd(fresh)
        self.names = get_static_names(MEME_CLF_VERSION)["names"]
        self.total = len(self.names)

    def train(self) -> None:
        self.begin = time()
        self.now = time()
        names_done = list(self.smd["name_acc"].keys())
        for name in filter(lambda x: x not in names_done, self.names):
            self.smd["name_acc"][name] = 0
            for cp in StonkTrainer(name, True).train(num_workers=8, batch_size=64):
                self.smd["name_acc"][name] = cp["max_val_acc"]
                break
            dump_smd(self.smd)
            self.num_done = len(self.smd["name_acc"])
            self.print_stats()

    def train_laggers(self, under: float) -> None:
        self.begin = time()
        self.now = time()
        for name in filter(
            lambda name: self.smd["name_acc"][name] < under, self.smd["name_acc"].keys()
        ):
            for cp in StonkTrainer(name, False).train(num_workers=8, batch_size=64):
                self.smd["name_acc"][name] = max(
                    cp["max_val_acc"], self.smd["name_acc"][name]
                )
                break
            dump_smd(self.smd)
            self.num_done = sum(
                1 if self.smd["name_acc"][name] >= under else 0
                for name in self.smd["name_acc"]
            )
            self.print_stats()

    def test_models(self) -> None:
        for name in list(self.smd["name_acc"].keys()):
            print(f"Loading data for {name}")
            trainer = StonkTrainer(name, fresh=False)
            if trainer.test_model() == "k":
                break

    def print_worst(self) -> None:
        display_df(
            pd.DataFrame(
                list(self.smd["name_acc"].items()), columns=["name", "acc"]
            ).sort_values("acc", ascending=True)
        )

    def summary(self) -> None:
        clear_output()
        print(f"name: MemeMarket")
        self.print_stats()

    def print_stats(self) -> None:
        uptime = int(time() - self.begin)
        round_time = int(time() - self.now)
        self.now = time()
        self.smd["total_time"] += round_time
        num_left = self.total - self.num_done
        pretty_print_dict(
            dict(
                num_names_done=f"{self.num_done}/{self.total}",
                round_time=secondsToText(round_time),
                market_uptime=secondsToText(uptime),
                timestamp=cast(str, arrow.utcnow().to("local").format("HH:mm:ss")),
                eta=secondsToText((self.smd["total_time"] * num_left) // self.num_done),
            )
        )
        self.histo()

    def histo(self):
        df = pd.DataFrame(
            list(self.smd["name_acc"].items()), columns=["name", "acc"]
        ).sort_values(by="acc", ascending=True)
        _ = plt.figure(figsize=(20, 6))
        _ = plt.hist(df["acc"], range=(0.7, 1), bins=3 * 40)
        plt.grid()
        plt.title("Market Accuracy Histogram")
        plt.show()
        df["acc"] = df["acc"].apply(round_3)
        display_df(df[:25])


def round_3(num: float):
    return round(num, 3)


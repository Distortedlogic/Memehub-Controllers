from time import time
from typing import cast

import arrow
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

    def train(self, test: bool = False) -> None:
        if input("Want fresh models?"):
            fresh = True
        else:
            fresh = False
        self.begin = time()
        self.now = time()
        names_done = list(self.smd["name_acc"].keys())
        keypress = ""
        for name in filter(lambda x: x not in names_done, self.names):
            self.smd["name_acc"][name] = 0
            trainer = StonkTrainer(name, fresh)
            for cp in trainer.train(num_workers=32, batch_size=64, num_epochs=2):
                self.smd["name_acc"][name] = max(
                    cp["max_val_acc"], self.smd["name_acc"][name]
                )
                break
            dump_smd(self.smd)
            self.print_stats()
            if test and keypress != "c" and (keypress := trainer.test_model()):
                if keypress == "k":
                    break

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
        num_left = self.total - len(self.smd["name_acc"])
        num_done = len(self.smd["name_acc"])
        pretty_print_dict(
            dict(
                num_names_done=f"{num_done}/{self.total}",
                worst=min(self.smd["name_acc"].values()),
                round_time=secondsToText(round_time),
                market_uptime=secondsToText(uptime),
                timestamp=cast(str, arrow.utcnow().to("local").format("HH:mm:ss")),
                eta=secondsToText((self.smd["total_time"] * num_left) // num_done),
            )
        )

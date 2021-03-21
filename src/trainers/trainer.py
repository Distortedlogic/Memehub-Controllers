import json
import time
from typing import Any, List, Tuple, cast

import arrow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.series import Series
from src.utils.data_folders import name_to_img_count
from src.utils.display import display_df
from src.utils.model_func import avg_n, check_point, get_static_names, load_cp
from src.utils.secondToText import secondsToText


class Trainer:
    def __init__(self, name: str, version: str):
        if input("Do you want fresh?") == "y":
            self.fresh = True
        else:
            self.fresh = False
        self.cp = load_cp(name, version, self.fresh)
        self.static = get_static_names(version)
        self.correct: int
        self.total: int
        self.patience: int
        self.num_epochs: int
        self.epoch: int
        self.begin: int
        self.model: Any

    def get_num_correct(self, is_validation: bool) -> Tuple[int, int]:
        raise NotImplementedError

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

    def display_wrong_names(self) -> None:
        df = pd.DataFrame(
            list(self.get_wrong_names().items()), columns=["name", "num_wrong"]
        )
        df = df[df["num_wrong"] != 0]
        df["img_count"] = cast(Series[str], df["name"]).apply(name_to_img_count)
        display_df(df.sort_values("num_wrong", ascending=False))

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
        if self.cp["iteration"] >= 100:
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

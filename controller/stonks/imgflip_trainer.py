from os import listdir
from os.path import splitext

import pickle, time
import pandas as pd

from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.optimizers import SGD
from keras.models import load_model

from controller.constants import MODELS_REPO
from controller.extensions import db
from controller.stonks.schema import NotMeme, TrainData

import tensorflow as tf

var_converter = tf.compat.v1.graph_util.convert_variables_to_constants

optimizer_kwargs = {
    "lr": 0.001,
    "clipnorm": 1.0,
    "momentum": 0.9,
    "decay": 0.0,
    "nesterov": True,
}
epochs = 2


def base_model():
    model = Sequential()
    model.add(Dense(units=16, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    optimizer_kwargs = {
        "lr": 0.001,
        "clipnorm": 1.0,
        "momentum": 0.9,
        "decay": 0.0,
        "nesterov": True,
    }
    model.compile(
        optimizer=SGD(**optimizer_kwargs),
        loss=binary_crossentropy,
        metrics=["accuracy"],
    )
    return model


class ImgflipTrainer:
    def get_not_memes(self):
        not_memes = db.session.query(NotMeme).statement
        not_memes = pd.read_sql(not_memes, db.session.bind)
        return pd.DataFrame(not_memes["features"].tolist()).append(
            not_memes["name"], axis=1, ignore_index=True
        )

    def run(self) -> None:
        num_per_name = 15
        names = db.session.query(TrainData.name).group_by(TrainData.name)
        self.df = pd.DataFrame()
        names_done = list(set(splitext(file)[0] for file in listdir(MODELS_REPO)))
        for (name,) in names:
            q = (
                db.session.query(TrainData)
                .filter(TrainData.name == name)
                .limit(num_per_name)
                .statement
            )
            self.df = self.df.append(pd.read_sql(q, db.session.bind), ignore_index=True)
        self.df = pd.DataFrame(self.df["features"].tolist()).append(
            self.df["name"], axis=1, ignore_index=True
        )
        self.df = self.df.append(self.get_not_memes(), ignore_index=True)
        for (name,) in names:
            if name not in names_done:
                self.train(name)

    def train(self, name) -> None:
        print(name)
        try:
            model = load_model(MODELS_REPO + f"{name}.pb")
        except:
            model = base_model()
        num_needed = int(0.25 * len(self.df)) + 1 - sum(self.df["name"] == name)
        q = db.session.query(TrainData).filter(TrainData.name == name).statement
        name_df = pd.read_sql(q, db.session.bind).sample(n=num_needed, replace=True)
        name_df = pd.DataFrame(name_df["features"].tolist()).append(
            name_df["name"], axis=1, ignore_index=True
        )
        features = (
            self.df.append(name_df, ignore_index=True)
            .sample(frac=1)
            .reset_index(drop=True)
        )
        names = (features.pop("name") == name).values
        model.fit(features.values, names, epochs=epochs, shuffle=True)
        ins = [node.op.name for node in model.inputs]
        outs = [node.op.name for node in model.outputs]
        name = name.replace("?", "")
        with open(MODELS_REPO + f"{name}.pkl", "wb") as cp_file:
            pickle.dump(dict(inputs=ins, outputs=outs), cp_file)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            frozen_graph = var_converter(sess, sess.graph_def, outs)
        tf.train.write_graph(frozen_graph, MODELS_REPO, f"{name}.pb", as_text=False)

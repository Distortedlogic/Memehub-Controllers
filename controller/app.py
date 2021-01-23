import numpy as np
from celery import Celery
from celery_singleton import Singleton
from decouple import config
from flask import Flask, jsonify, request
from redisai import Client

from controller.generated.models import db
from controller.utils.image_funcs import extract_img

num_to_name = {}


def create_celery_app(app=None):
    app = app or create_app()
    celery = Celery(
        app.import_name,
        broker=app.config["CELERY_BROKER_URL"],
        include=["controller.tasks"],
    )
    celery.conf.update(app.config)

    class ContextTask(Singleton):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return Singleton.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object("config.flask")
    db.init_app(app)
    rai = Client(host="redis", port="6379")

    @app.route("/meme_clf", methods=["POST"])
    def meme_clf():
        rai.tensorset(
            "image", np.array([extract_img(request["data"]["url"])]).astype(np.float32),
        )
        rai.modelrun("meme_features", ["image"], ["features"])
        return jsonify({"name": num_to_name[rai.tensorget("out")[0]]})

    return app

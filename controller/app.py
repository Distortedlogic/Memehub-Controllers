import json

import numpy as np
from celery_singleton import Singleton
from flask import Flask, jsonify, request
from flask_cors import CORS
from redisai import Client

from celery import Celery
from controller.constants import STATIC_PATH
from controller.generated.models import db
from controller.utils.model_func import load_img_from_url

TASK_LIST = ["controller.reddit.tasks"]


def create_celery_app(app=None):
    app = app or create_app()
    celery = Celery(
        app.import_name, broker=app.config["CELERY_BROKER_URL"], include=TASK_LIST,
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
    CORS(app)
    app.config.from_object("config.flask")
    db.init_app(app)
    rai = Client(host="redis", port="6379")
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)

    @app.route("/meme_clf", methods=["POST"])
    def meme_clf():
        images = [load_img_from_url()(request.json["url"])]
        rai.tensorset("image", np.array(images, dtype=np.float32))
        rai.modelrun("MemeClf", ["image"], ["out"])
        pred = static["num_name"][str(rai.tensorget("out")[0])]
        return jsonify(dict(pred=pred))

    return app

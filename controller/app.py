import json
from typing import Any, Dict, Union, cast

import numpy as np
from celery import Celery
from celery_singleton import Singleton
from flask import Flask, jsonify, request
from flask_cors import CORS
from redisai import Client

from controller.constants import STATIC_PATH
from controller.generated.models import db
from controller.utils.model_func import load_img_from_url

TASK_LIST = ["controller.reddit.tasks"]


def create_celery_app(app: Union[Flask, None] = None):
    app = app or create_app()
    celery = Celery(
        app.import_name, broker=app.config["CELERY_BROKER_URL"], include=TASK_LIST,
    )
    celery.conf.update(app.config)

    class ContextTask(Singleton):
        abstract = True

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            with cast(Flask, app).app_context():
                return cast(Any, Singleton.__call__(self, *args, **kwargs))

    celery.Task = ContextTask
    return celery


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    _ = CORS(app)
    app.config.from_object("config.flask")
    db.init_app(app)
    rai = Client(host="redis", port=6379)
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)

    @app.route("/meme_clf", methods=["POST"])
    def meme_clf() -> Dict[str, Any]:
        images = [load_img_from_url(cast(str, request.json["url"]))]
        _ = rai.tensorset("image", np.array(images, dtype=np.float32))
        _ = rai.modelrun("MemeClf", ["image"], ["out"])
        pred = static["num_name"][str(cast(int, rai.tensorget("out")[0]))]
        return cast(Dict[str, Any], jsonify(dict(pred=pred)))

    return app

from typing import Any, Dict, cast

import numpy as np
from flask_cors import CORS
from redisai import Client

from flask import Flask, jsonify, request
from src.constants import LOAD_VERSION
from src.utils.image_funcs import load_img_from_url
from src.utils.model_func import get_static_names


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    _ = CORS(app)
    app.config.from_object("config.flask")
    rai = Client(host="redis", port=6379)
    num_name = get_static_names(LOAD_VERSION)["num_name"]

    @app.route("/meme_clf", methods=["POST"])
    def meme_clf() -> Dict[str, Any]:
        images = [load_img_from_url(cast(str, request.json["url"]))]
        _ = rai.tensorset("image", np.array(images, dtype=np.float32))
        _ = rai.modelrun("MemeClf", ["image"], ["out"])
        pred = num_name[str(cast(int, rai.tensorget("out")[0]))]
        return cast(Dict[str, Any], jsonify(dict(pred=pred)))

    return app

from typing import Any, Dict, cast
from uuid import uuid4

import numpy as np
from flask_cors import CORS
from redisai import Client

from flask import Flask, jsonify, request
from src.constants import LOAD_MEME_CLF_VERSION
from src.utils.image_funcs import load_tensor_from_url
from src.utils.model_func import get_static_names


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    _ = CORS(app)
    app.config.from_object("config.flask")
    rai = Client(host="redis", port=6379)
    num_name = get_static_names(LOAD_MEME_CLF_VERSION)["num_name"]

    @app.route("/meme_clf", methods=["POST"])
    def meme_clf() -> Dict[str, Any]:
        images = [
            load_tensor_from_url(cast(str, request.json["url"]), is_deleted=False)
        ]
        _ = rai.tensorset(hash := str(uuid4()), np.array(images, dtype=np.float32))
        _ = rai.modelrun("features", hash, hash)
        _ = rai.modelrun("dense", hash, hash)
        pred = num_name[str(cast(int, rai.tensorget(hash)[0]))]
        _ = rai.delete(hash)
        return cast(Dict[str, Any], jsonify(dict(pred=pred)))

    return app

from typing import List, cast

import numpy as np
import redisai
from sqlalchemy import or_
from sqlalchemy.sql.elements import ClauseElement
from src.constants import LOAD_VERSION
from src.generated.models import Meme
from src.session import site_db
from src.utils.image_funcs import load_img_from_url
from src.utils.model_func import get_static_names


def update():
    rai = redisai.Client(host="redis", port=6379)
    num_name = get_static_names(LOAD_VERSION)["num_name"]
    while memes := (
        site_db.query(Meme)
        .filter(
            or_(
                cast(ClauseElement, Meme.stonk == None),
                cast(ClauseElement, Meme.version != LOAD_VERSION),
            )
        )
        .order_by(Meme.createdAt.desc())
        .limit(100)
        .all()
    ):
        images = [load_img_from_url(meme.url) for meme in memes]
        _ = rai.tensorset("images", np.array(images, dtype=np.float32))
        _ = rai.modelrun("meme_clf", ["images"], ["out"])
        names = (num_name[str(num)] for num in cast(List[int], rai.tensorget("out")))
        for meme, name in zip(memes, names):
            meme.stonk = name
            meme.version = VERSION  # type: ignore

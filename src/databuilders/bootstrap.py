import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, cast

from sqlalchemy.sql.elements import ClauseElement
from src.constants import INCORRECT_REPO, MEMES_REPO, NOT_TEMPLATE_REPO
from src.generated.models import RedditMeme
from src.session import site_db
from src.utils.image_funcs import download_img_from_url
from tqdm import tqdm


def engine(meme: RedditMeme) -> None:
    repo = MEMES_REPO if meme.stonk_correct else INCORRECT_REPO
    filename = meme.url.split("/")[-1]
    folder = repo + meme.meme_clf
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = folder + "/" + filename
    _ = download_img_from_url(meme.url, path)
    if not meme.is_a_template:
        _ = download_img_from_url(meme.url, NOT_TEMPLATE_REPO + filename)


def bootstrap_data() -> None:
    q = site_db.query(RedditMeme).filter(
        cast(ClauseElement, RedditMeme.stonk_correct != None)
    )
    with Pool(os.cpu_count()) as workers:
        _: List[None] = list(tqdm(workers.imap_unordered(engine, q), total=q.count()))

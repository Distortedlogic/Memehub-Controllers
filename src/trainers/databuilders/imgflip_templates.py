import os
import shutil
from itertools import chain, repeat
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Iterable, Iterator, List, Tuple, cast

import boto3
import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import ResultSet
from decouple import config
from pathlib2 import Path
from sqlalchemy.sql.elements import ClauseElement
from src.constants import (ALT_NAMES, BLANKS_REPO, DONT_USE_TEMPLATES,
                           IMGFLIP_TEMPALTE_URI, MEMES_TO_USE,
                           REPLICATED_TEMPLATES_GROUPED, USER_TEMPLATES)
from src.schema import Template
from src.session import training_db
from src.utils.aws_s3 import upload_to_aws
from src.utils.display import display_df
from src.utils.image_funcs import download_img_from_url
from src.utils.sanitize import sanitize_template_name
from tqdm import tqdm

s3 = boto3.resource(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY")
)
bucket = s3.Bucket("memehub")


def get_template_data(page_number: int) -> List[Dict[str, str]]:
    with requests.get(IMGFLIP_TEMPALTE_URI.format(page_number)) as resp:
        soup = BeautifulSoup(resp.text, "lxml")
        templates = USER_TEMPLATES
    for meme in cast(Iterator[ResultSet], soup.find_all("div", class_="mt-img-wrap")):
        name: str = meme.a["title"][:-5]  # type: ignore
        page: str = "https://imgflip.com" + meme.a["href"] + "?page={}"  # type: ignore
        blank_url: str = "https:" + meme.a.img["src"]  # type: ignore
        if (name := sanitize_template_name(name)) not in DONT_USE_TEMPLATES:
            if name in ALT_NAMES:
                for group in REPLICATED_TEMPLATES_GROUPED:
                    if name in group:
                        name = group[-1]
            if name in MEMES_TO_USE:
                templates.append(dict(name=name, page=page, blank_url=blank_url))
    return templates


def build_db() -> None:
    print(f"num memes to use - {len(MEMES_TO_USE)}")
    _: Any = training_db.query(Template).delete()
    training_db.commit()
    step_size = 50
    pages = range(0, step_size)
    with Pool(cpu_count() // 2) as workers:
        df = pd.DataFrame.from_records(
            list(
                chain.from_iterable(
                    cast(
                        Iterable[List[Dict[str, str]]],
                        tqdm(
                            workers.imap_unordered(get_template_data, pages),
                            total=step_size,
                        ),
                    )
                )
            )
        ).drop_duplicates(ignore_index=True)
    df.to_sql("templates", training_db.bind, if_exists="replace", index_label="id")
    print(f"Total Templates - {training_db.query(Template).count()}")


def download_img_from_url_blank(name_blank_path: Tuple[Tuple[str, str], str]) -> bool:
    (name, blank), path = name_blank_path
    if not (
        success := download_img_from_url(
            blank, path=path + name + os.path.splitext(blank)[1], verbose=True
        )
    ):
        print(blank)
        return success
    else:
        return success


def download_blanks() -> None:
    q = training_db.query(Template.name, Template.blank_url)
    path = BLANKS_REPO
    shutil.rmtree(path)
    Path(path).mkdir(parents=True, exist_ok=True)  # type: ignore
    with Pool(cpu_count()) as workers:
        results: List[bool] = list(
            tqdm(
                workers.imap_unordered(
                    download_img_from_url_blank, zip(q, repeat(path))
                ),
                total=q.count(),
            )
        )
    print(f"{sum(results)}/{len(results)}")


def upload_to_aws_mp(filename: str) -> bool:
    return upload_to_aws(
        BLANKS_REPO + filename,
        "memehub/templates/" + os.path.splitext(filename)[0],
        False,
    )


def blanks_to_aws() -> None:
    bucket.objects.filter(Prefix="memehub/templates/").delete()
    filenames = list(os.listdir(BLANKS_REPO))
    with Pool(cpu_count() // 2) as workers:
        results: List[bool] = list(
            tqdm(
                workers.imap_unordered(upload_to_aws_mp, filenames),
                total=len(filenames),
            )
        )
    print(f"{sum(results)}/{len(results)}")


def miss_match() -> None:
    downloaded_blanks = [
        os.path.splitext(filename)[0] for filename in os.listdir(BLANKS_REPO)
    ]
    display_df(
        pd.read_sql(
            cast(
                str,
                training_db.query(Template)
                .filter(cast(ClauseElement, ~Template.name.in_(downloaded_blanks)))
                .statement,
            ),
            training_db.bind,
        )
    )

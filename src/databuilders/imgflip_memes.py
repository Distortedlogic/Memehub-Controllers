import os
import shutil
from itertools import chain
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, cast

import pandas as pd
import requests
from bs4 import BeautifulSoup
from retry import retry
from sqlalchemy.sql.elements import ClauseElement
from src.constants import (
    MEMES_REPO,
    MEMES_TO_USE,
    NOT_MEME_REPO,
    NOT_TEMPLATE_REPO,
    REPLICATED_TEMPLATES_GROUPED,
)
from src.schema import (
    MemeCorrectTest,
    MemeCorrectTrain,
    MemeIncorrectTest,
    MemeIncorrectTrain,
    NotAMemeTest,
    NotAMemeTrain,
    NotATemplateTest,
    NotATemplateTrain,
    Template,
)
from src.session import training_db
from src.utils.display import display_df
from src.utils.image_funcs import download_img_from_url
from tqdm import tqdm

num_pages = 100


@retry(tries=5, delay=2)
def img_urls_from_page(page: str) -> List[str]:
    with requests.get(page) as resp:
        soup = BeautifulSoup(resp.text, "lxml")
    meme_containers = soup.find_all("img", class_="base-img")
    return [
        "https:" + meme["src"] for meme in cast(List[Dict[str, str]], meme_containers)
    ]


def engine_mp(name_page: Tuple[str, str]) -> None:
    name, page = name_page
    urls = list(
        chain.from_iterable(
            img_urls_from_page(page.format(i)) for i in range(1, num_pages)
        )
    )
    counter = len(os.listdir(MEMES_REPO + name))
    while (counter := counter + 1) <= 1000 and urls:
        url = urls.pop()
        filename = url.split("/")[-1]
        path = MEMES_REPO + f"{name}/{filename}"
        _ = download_img_from_url(url, path)


def download_imgflip_memes(fresh: bool = False):
    if not training_db.query(Template).count():
        raise Exception("No Templates")
    for (name,) in training_db.query(Template.name):
        Path(MEMES_REPO + name).mkdir(parents=True, exist_ok=True)
    if fresh:
        name_page = training_db.query(Template.name, Template.page).all()
    else:
        names_filled = list(
            name
            for name in os.listdir(MEMES_REPO)
            if len(list(os.listdir(MEMES_REPO + name))) >= 1000
        )
        name_page = (
            training_db.query(Template.name, Template.page)
            .filter(cast(ClauseElement, ~Template.name.in_(names_filled)))
            .all()
        )
    with Pool(cpu_count()) as workers:
        _: List[None] = list(
            tqdm(workers.imap_unordered(engine_mp, name_page), total=len(name_page))
        )


def print_names_missed():
    print(
        list(
            name
            for name, in training_db.query(Template.name).filter(
                cast(ClauseElement, ~Template.name.in_(list(os.listdir(MEMES_REPO))))
            )
        )
    )


def fill_missing():
    name_page = (
        training_db.query(Template.name, Template.page)
        .filter(cast(ClauseElement, ~Template.name.in_(list(os.listdir(MEMES_REPO)))))
        .all()
    )
    _: List[None] = list(tqdm(map(engine_mp, name_page), total=len(name_page)))


def print_num_memes_in_folders():
    display_df(
        pd.DataFrame.from_records(
            list(
                {"name": name, "num": len(list(os.listdir(MEMES_REPO + name)))}
                for name in MEMES_TO_USE
            )
        ).sort_values("num")
    )


def fill_folders():
    names_to_fill = list(
        name
        for name in os.listdir(MEMES_REPO)
        if len(list(os.listdir(MEMES_REPO + name))) < 100
    )
    name_page = (
        training_db.query(Template.name, Template.page)
        .filter(cast(ClauseElement, Template.name.in_(names_to_fill)))
        .all()
    )
    print(f"names_to_fill = {len(names_to_fill)}")
    _: List[None] = list(tqdm(map(engine_mp, name_page), total=len(name_page)))
    # with Pool(cpu_count()) as workers:
    #     list(tqdm(workers.imap_unordered(engine_mp, name_page), total=len(name_page)))


def merge_replicated_folder():
    bad_folders: List[str] = []
    for names in cast(
        List[List[str]],
        tqdm(REPLICATED_TEMPLATES_GROUPED, total=len(REPLICATED_TEMPLATES_GROUPED)),
    ):
        if not os.path.isdir(dest := MEMES_REPO + names[-1] + "/"):
            os.makedirs(dest)
        for name in names[:-1]:
            name_dir = MEMES_REPO + name + "/"
            if os.path.isdir(name_dir):
                for filename in os.listdir(name_dir):
                    try:
                        _ = shutil.copyfile(name_dir + filename, dest + filename)
                    except shutil.SameFileError:
                        pass
            else:
                bad_folders.append(name)
    print("bad folders", bad_folders)


def name_imgs_to_db(name: str):
    for idx, filename in enumerate(os.listdir(MEMES_REPO + f"{name}")):
        path = MEMES_REPO + f"{name}/{filename}"
        if idx % 10 == 0:
            training_db.add(MemeCorrectTest(name=name, path=path))
        else:
            training_db.add(MemeCorrectTrain(name=name, path=path))
    training_db.commit()


def add_not_meme(idx_filename: Tuple[int, str]):
    idx, filename = idx_filename
    if idx % 10 == 0:
        training_db.add(NotAMemeTest(path=NOT_MEME_REPO + filename))
    else:
        training_db.add(NotAMemeTrain(path=NOT_MEME_REPO + filename))
    training_db.commit()


def add_not_template(idx_filename: Tuple[int, str]):
    idx, filename = idx_filename
    if idx % 10 == 0:
        training_db.add(NotATemplateTest(path=NOT_TEMPLATE_REPO + filename))
    else:
        training_db.add(NotATemplateTrain(path=NOT_TEMPLATE_REPO + filename))
    training_db.commit()


def meme_add_name_idx(name: str):
    for idx, meme in enumerate(
        training_db.query(MemeCorrectTrain).filter(
            cast(ClauseElement, MemeCorrectTrain.name == name)
        )
    ):
        meme.name_idx = idx  # type:ignore
    for idx, meme in enumerate(
        training_db.query(MemeCorrectTest).filter(
            cast(ClauseElement, MemeCorrectTest.name == name)
        )
    ):
        meme.name_idx = idx  # type:ignore
    for idx, meme in enumerate(
        training_db.query(MemeIncorrectTrain).filter(
            cast(ClauseElement, MemeIncorrectTrain.name == name)
        )
    ):
        meme.name_idx = idx  # type:ignore
    for idx, meme in enumerate(
        training_db.query(MemeIncorrectTest).filter(
            cast(ClauseElement, MemeIncorrectTest.name == name)
        )
    ):
        meme.name_idx = idx  # type:ignore
    training_db.commit()


def add_name_idx(entity: Any):
    for idx, meme in enumerate(training_db.query(entity)):
        meme.name_idx = idx
    training_db.commit()


def build_db_from_imgdir():
    _: Any = training_db.query(MemeCorrectTrain).delete()
    _: Any = training_db.query(MemeCorrectTest).delete()
    _: Any = training_db.query(NotAMemeTrain).delete()
    _: Any = training_db.query(NotAMemeTest).delete()
    _: Any = training_db.query(NotATemplateTrain).delete()
    _: Any = training_db.query(NotATemplateTest).delete()
    training_db.commit()
    template_names = list(
        cast(Set[str], set.intersection(set(os.listdir(MEMES_REPO)), MEMES_TO_USE))
    )
    files = list(os.listdir(NOT_MEME_REPO))
    with Pool(cpu_count()) as workers:
        _ = list(
            tqdm(
                workers.imap_unordered(add_not_meme, enumerate(files)), total=len(files)
            )
        )
    files = list(os.listdir(NOT_TEMPLATE_REPO))
    with Pool(cpu_count()) as workers:
        _ = list(
            tqdm(
                workers.imap_unordered(add_not_template, enumerate(files)),
                total=len(files),
            )
        )
    with Pool(cpu_count()) as workers:
        _ = list(
            tqdm(
                workers.imap_unordered(name_imgs_to_db, template_names),
                total=len(template_names),
            )
        )
    q_list: List[Any] = [
        NotAMemeTrain,
        NotAMemeTest,
        NotATemplateTrain,
        NotATemplateTest,
    ]
    with Pool(cpu_count()) as workers:
        _ = list(tqdm(workers.imap_unordered(add_name_idx, q_list), total=len(q_list)))
    with Pool(cpu_count()) as workers:
        _ = list(
            tqdm(
                workers.imap_unordered(meme_add_name_idx, template_names),
                total=len(template_names),
            )
        )


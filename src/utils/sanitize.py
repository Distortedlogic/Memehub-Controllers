import json
import os
import shutil
from multiprocessing import Pool, cpu_count
from typing import Any, List, Tuple, cast

from sqlalchemy.sql.elements import ClauseElement
from sqlalchemy.sql.schema import Column
from src.constants import (
    ALT_NAMES,
    DONT_USE_TEMPLATES,
    MEMES_REPO,
    MEMES_TO_USE,
    NOT_MEME_REPO,
    REPLICATED_TEMPLATES_GROUPED,
    SEEN_MEMES,
)
from src.generated.models import RedditMeme
from src.session import site_db
from tqdm import tqdm


def sanitize_template_name(name: str) -> str:
    return (
        "".join(
            i for i in name.encode("ascii", "ignore").decode() if i not in ":*?<>|.'(),"
        )
        .strip()
        .lower()
    )


def delete_unused_folders():
    for name in list(os.listdir(MEMES_REPO)):
        if name not in MEMES_TO_USE + list(ALT_NAMES):
            shutil.rmtree(MEMES_REPO + name + "/")


def copy_folder(name: str):
    src = MEMES_REPO + name
    dest = MEMES_REPO + sanitize_template_name(name)
    if src != dest:
        try:
            os.rename(src, dest)
        except Exception:
            for file in os.listdir(src):
                try:
                    shutil.copy(src + file, dest + file)
                except Exception:
                    pass


def rename_folder():
    folders = list(os.listdir(MEMES_REPO))
    with Pool(cpu_count()) as workers:
        _: Any = list(
            tqdm(workers.imap_unordered(copy_folder, folders), total=len(folders))
        )


def fix_memes(name: str):
    for meme in site_db.query(RedditMeme).filter(
        cast(ClauseElement, RedditMeme.meme_clf == name)
    ):
        meme.meme_clf = cast(Column[str], sanitize_template_name(meme.meme_clf))
    site_db.commit()


def sanitize_db():
    names_to_fix = [
        name
        for name, in cast(
            List[Tuple[str]],
            site_db.query(RedditMeme.meme_clf).distinct(RedditMeme.meme_clf),
        )
        if name != None and name != sanitize_template_name(name)
    ]
    with Pool(cpu_count()) as workers:
        _: Any = list(
            tqdm(
                workers.imap_unordered(fix_memes, names_to_fix), total=len(names_to_fix)
            )
        )


def sanitize_imgnet():
    NEW_REPO = "src/data/new_not_a_meme/"
    for file in os.listdir(NOT_MEME_REPO):
        filename, ext = os.path.splitext(file)
        ext = ext.lower()
        new_name = "".join(
            char for char in sanitize_template_name(filename) if char != " "
        )
        src = NOT_MEME_REPO + file
        dest = NEW_REPO + new_name + ext
        if ext in [".jpg", ".jpeg", ".jpe", ".jif", ".jfif", ".png", ".PNG"]:
            try:
                shutil.copy(src, dest)
                os.remove(src)
            except Exception:
                pass


def fix_constants():
    DONT_USE = set()
    for name in DONT_USE_TEMPLATES:
        DONT_USE.add(sanitize_template_name(name))

    REPLICATED_GROUP: List[List[str]] = []
    for group in REPLICATED_TEMPLATES_GROUPED:
        new_group: List[str] = []
        for name in group:
            new_group.append(sanitize_template_name(name))
        REPLICATED_GROUP.append(new_group)

    SEEN: List[str] = []
    for name in SEEN_MEMES:
        SEEN.append(sanitize_template_name(name))

    with open("DONT_USE.json", "w") as f:
        json.dump(dict(data=list(DONT_USE)), f)

    with open("REPLICATED_GROUP.json", "w") as f:
        json.dump(dict(data=REPLICATED_GROUP), f)

    with open("SEEN.json", "w") as f:
        json.dump(dict(data=SEEN), f)

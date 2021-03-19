import os
import shutil
from typing import List

from src.constants import MEMES_REPO


def name_to_img_count(name: str) -> int:
    return len(list(os.listdir(MEMES_REPO + name)))


def merge_folders(names: List[str]):
    if not os.path.isdir(dest := MEMES_REPO + names[-1] + "/"):
        os.makedirs(dest)
    for name in names[:-1]:
        name_dir = MEMES_REPO + name + "/"
        if os.path.isdir(name_dir):
            print(name_dir)
            for filename in os.listdir(name_dir):
                print(filename)
                _ = shutil.copyfile(name_dir + filename, dest + filename)

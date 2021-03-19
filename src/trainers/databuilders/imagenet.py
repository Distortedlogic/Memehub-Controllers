import json
import os
import time
from typing import Dict

import requests
from redisai import Client
from src.constants import NOT_MEME_REPO
from src.utils.image_funcs import download_img_from_url
from src.utils.sanitize import sanitize_template_name
from src.utils.secondToText import secondsToText

WNID_TO_URLS = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="
starting_info = "src/data/imagenet/imagenet_class_info.json"
good_urls = "src/data/imagenet/good_urls.json"
to_do_json = "src/data/imagenet/to_do.json"
bad_names_json = "src/data/imagenet/bad_names.json"

rai = Client(host="127.0.0.1", port=6379)


def get_to_do(fresh: bool) -> Dict[str, str]:
    if fresh:
        with open(starting_info) as f:
            to_do = json.load(f)
    else:
        with open(to_do_json) as f:
            to_do = json.load(f)
    with open(bad_names_json) as f:
        bad_names = json.load(f)["data"]
    for name in bad_names:
        try:
            del to_do[name]
        except:
            pass
    with open(to_do_json, "w", encoding="utf-8") as f:
        json.dump(to_do, f, ensure_ascii=False, indent=4)
    return to_do


def imagenet_db_build(fresh: bool = False) -> None:
    to_do = get_to_do(fresh)
    total = len(to_do)
    start = time.time()
    for idx, (name, class_wnid) in enumerate(to_do.copy().items()):
        while True:
            try:
                resp = requests.get(WNID_TO_URLS + class_wnid, timeout=60)
                break
            except:
                time.sleep(5)
        print(resp.content)
        raise Exception("check")
        for url in resp.content.splitlines():
            url = url.decode("utf-8")
            filename, ext = os.path.splitext(url.split("/")[-1])
            print("url")
            print(url)
            if ext in ["jpg", "jpeg", "jpe", "jif", "jfif", "png", "PNG"]:
                path = (
                    NOT_MEME_REPO
                    + sanitize_template_name(name + "_" + filename)
                    + "."
                    + ext
                )
                if download_img_from_url(url, path):
                    break
        raise Exception("check")
        del to_do[name]
        with open(to_do_json, "w", encoding="utf-8") as f:
            json.dump(to_do, f, ensure_ascii=False, indent=4)

        # clear_output()
        print(name)
        uptime = int(time.time() - start)
        num_left = total - idx
        print(f"idx - {idx}")
        print(f"num left - {num_left}")
        print(f"uptime - {secondsToText(uptime)}")
        print(f"ETA - {secondsToText((uptime*num_left)//(idx+1))}")

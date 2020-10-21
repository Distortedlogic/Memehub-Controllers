from socket import timeout
from controller.stonks.schema import NotMeme
from controller.utils.extract_image import extract_img
from controller.constants import IMAGENET_REPO
from controller.extensions import db
import json, os, redisai, time
import numpy as np
import pandas as pd
from functools import partial
from itertools import chain
from multiprocessing import Pool

import requests
from billiard import Pool, cpu_count
from tqdm import tqdm

from controller.utils.save_image import save_img

from sqlalchemy import func

clear = lambda: os.system("clear")

IMAGENET_API_WNID_TO_URLS = (
    lambda wnid: f"http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={wnid}"
)
starting_info = "controller/stonks/not_a_meme/imagenet/imagenet_class_info.json"
good_urls = "controller/stonks/not_a_meme/imagenet/good_urls.json"
to_do = "controller/stonks/not_a_meme/imagenet/to_do.json"
bad_names = "controller/stonks/not_a_meme/imagenet/bad_names.json"


class ImageNet:
    def __init__(self):
        self.rai = redisai.Client(host="redis", port="6379")
        with open(to_do) as f:
            self.to_do = json.load(f)
        if not self.to_do:
            with open(starting_info) as f:
                self.to_do = json.load(f)
        with open(good_urls) as f:
            self.good_urls = json.load(f)
        with open(bad_names) as f:
            self.bad_names = json.load(f)["data"]

        # dups = db.session.query(NotMeme).group_by(NotMeme.name).having(func.count(NotMeme.name) > 1).all()
        # dups.delete()
        # print(dups)
        # raise Exception('halt')

    def get_names_images(self):
        # for name in set(chain.from_iterable(db.session.query(NotMeme.name).group_by(NotMeme.name))):
        #     try:
        #         del self.to_do[name]
        #     except:
        #         pass
        # with open(to_do, 'w', encoding='utf-8') as f:
        #     json.dump(self.to_do, f, ensure_ascii=False, indent=4)

        # for name in self.bad_names:
        #     try:
        #         del self.to_do[name]
        #     except:
        #         pass
        # with open(to_do, 'w', encoding='utf-8') as f:
        #     json.dump(self.to_do, f, ensure_ascii=False, indent=4)
        images = []
        names = []
        length = len(self.to_do)
        for idx, (name, class_wnid) in enumerate(self.to_do.copy().items()):
            while True:
                try:
                    resp = requests.get(
                        IMAGENET_API_WNID_TO_URLS(class_wnid), timeout=60
                    )
                    break
                except:
                    time.sleep(5)
            urls = resp.content.splitlines()
            tries = 0
            for url in urls:
                url = url.decode("utf-8")
                image = extract_img(url)
                if image.any():
                    images.append(image)
                    names.append(name)
                    self.good_urls[name] = url
                    with open(good_urls, "w", encoding="utf-8") as f:
                        json.dump(self.good_urls, f, ensure_ascii=False, indent=4)
                    print(f"{name} - {len(images)}")
                    break
                else:
                    tries += 1
                    if tries == len(urls):
                        self.bad_names.append(name)
                        with open(bad_names, "w", encoding="utf-8") as f:
                            json.dump(
                                dict(data=self.bad_names),
                                f,
                                ensure_ascii=False,
                                indent=4,
                            )
                        del self.to_do[name]
                        with open(to_do, "w", encoding="utf-8") as f:
                            json.dump(self.to_do, f, ensure_ascii=False, indent=4)
            if len(images) == 100 or idx == length:
                yield names, images
                images = []
                names = []

    def run(self):
        now = time.time()
        for names, images in self.get_names_images():
            model_runtime = time.time()
            print("model run")
            self.rai.tensorset("images", np.array(images).astype(np.float32))
            self.rai.modelrun("vgg16", ["images"], ["out"])
            print("exec time", time.time() - model_runtime)
            db.session.add_all(
                NotMeme(name=name, features=features.flatten().tolist())
                for name, features in zip(names, self.rai.tensorget("out"))
            )
            db.session.commit()
            for name in names:
                del self.to_do[name]
            with open(to_do, "w", encoding="utf-8") as f:
                json.dump(self.to_do, f, ensure_ascii=False, indent=4)
            print(f"there are {len(self.to_do)} left")
            print("full round runtime", now - time.time())
            now = time.time()

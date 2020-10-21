from itertools import chain
from multiprocessing import Pool

import requests
from bs4 import BeautifulSoup

from controller.extensions import db
from controller.stonks.schema import Template

IMG_FLIP_URI = "https://imgflip.com/memetemplates?page={}"
NUM_WORKERS = 8


class ImgflipTemplates:
    def __init__(self):
        self.step_size = 4

    def get_template_data(self, page_number: int):
        with requests.get(IMG_FLIP_URI.format(page_number)) as page:
            soup = BeautifulSoup(page.text, "lxml")
        memes = soup.find_all("div", class_="mt-box")
        if not memes:
            self.found_empty_page = True
        return [
            {
                "name": meme.div.a["title"][:-5],
                "page": "https://imgflip.com" + meme.div.a["href"],
            }
            for meme in memes
        ]

    def build_db(self) -> None:
        start_page = 1
        self.found_empty_page = False
        while not self.found_empty_page:
            end_page = start_page + self.step_size
            pages = range(start_page, end_page)
            with Pool(NUM_WORKERS) as workers:
                raw_batch = list(workers.imap_unordered(self.get_template_data, pages))
            start_page = end_page
            for template in chain.from_iterable(raw_batch):
                try:
                    db.session.query(Template).filter_by(name=template["name"]).one()
                except:
                    pass
                    db.session.add(Template(**template))
                    db.session.commit()

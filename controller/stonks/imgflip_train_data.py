import redisai, time

import numpy as np
import requests
from bs4 import BeautifulSoup

# from keras_preprocessing.image.image_data_generator import ImageDataGenerator

from controller.extensions import db
from controller.utils.extract_image import extract_img
from controller.stonks.schema import Template, TrainData
from controller.stonks.imgflip_templates import ImgflipTemplates

num_batches = 100


class ImgflipTrainData:
    def __init__(self):
        self.rai = redisai.Client(host="redis", port="6379")
        self.step_size = 4
        # self.datagen = ImageDataGenerator(
        #     rotation_range = 40,
        #     width_shift_range = 0.2,
        #     height_shift_range = 0.2,
        #     shear_range = 0.2,
        #     zoom_range = 0.2,
        #     horizontal_flip = True,
        #     fill_mode = 'nearest'
        # )

    def predict(self):
        self.rai.modelrun("vgg16", "images", "features")
        return self.rai.tensorget("features")

    def extract_imgs(self, page: str):
        urls = []
        for i in range(8):
            with requests.get(page.format(i)) as resp:
                meme_containers = BeautifulSoup(resp.text, "lxml").find_all(
                    "img", class_="base-img"
                )
            urls += ["https:" + meme["src"] for meme in meme_containers]
        images = []
        for url in urls:
            image = extract_img(url)
            if image.any():
                images.append(image)
        return np.array(images)

    def run(self):
        train_names = db.session.query(TrainData.name).group_by(TrainData.name).all()
        templates = (
            db.session.query(Template).filter(~Template.name.in_(train_names)).all()
        )
        if not train_names and not templates:
            ImgflipTemplates().build_db()
            templates = db.session.query(Template).all()
        elif not templates:
            raise Exception("Done")
        for name, page in ((template.name, template.page) for template in templates):
            print(f"extracting images {name}")
            now = time.time()
            imgs = self.extract_imgs(page)
            try:
                print(imgs.shape)
                assert imgs.shape[1:] == (224, 224, 3)
            except:
                print("bad shape")
            else:
                # transformer = self.datagen.flow(imgs)
                # batch = np.concatenate([imgs, np.concatenate([next(transformer) for _ in range(num_batches)])])
                self.rai.tensorset("images", imgs.astype(np.float32))
                db.session.add_all(
                    TrainData(name=name, features=features.flatten().tolist())
                    for features in self.predict()
                )
                db.session.commit()
                print("round runtime", time.time() - now)

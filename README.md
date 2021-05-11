# Memehub-Python

The datascience arm of https://memehub.lol

100% a personal project. Build as I go for fun, hence no proper docs on this project. But illistrates my use of technologies.

Is not stand alone, all memehub repos share a docker-compose file for services such as postgres and redis, which this repo depends.

Basic Type-Checking enforced through entire repo via pyright through pylance (see pyrightconfig.json file for settings)

### What is in this repo?

Reddit Meme Data ETL ran executed by Celery with rabbitmq messenger and redis backend.

ETL pipeline to build training/testing datsets for meme template classification via scraping imgflip.

Pytorch training of 400+ meme template computer vision models served on redisai.

Custom CLI conjioned with custom shell file to load models into redisai on docker container start.

Model auditing system that also bootstraps datasets for next version models.

Celery task for downloading images from data and running through the computer vision system. 

Custom statiscal ranking method of reddit users in meme related subreddits.

Percentile ranking of memes over 24 hour windows for meme investing and shorting.

Use of pytesseract OCR for meme text extraction. (NLP in the future for non-static meme template memes)

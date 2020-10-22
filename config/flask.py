from celery.schedules import crontab
from decouple import config

DEBUG = False
SECRET_KEY = config("SECRET_KEY")
CELERY_BROKER_URL = config("REDIS_URL")
CELERY_RESULT_BACKEND = config("REDIS_URL")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_REDIS_MAX_CONNECTIONS = 5
CELERYBEAT_SCHEDULE = {
    "reddit_memes_scrapper": {
        "task": "Reddit",
        "schedule": crontab(minute=0, hour="*"),
    },
}

SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_DATABASE_URI = config("DATABASE_URL")

MEMEDATA_URI = f"postgresql://postgres:postgres@memedata:5432/postgres"

SQLALCHEMY_BINDS = {"memedata": MEMEDATA_URI, "sitedata": config("DATABASE_URL")}

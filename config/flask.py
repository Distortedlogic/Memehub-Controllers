import os

from celery.schedules import crontab
from decouple import config
from get_docker_secret import get_docker_secret

SECRET_KEY = get_docker_secret(
    "SECRET_KEY", default=config("SECRET_KEY"), autocast_name=False
)
CELERY_BROKER_URL = "redis://redis:6379"
result_backend = "redis://redis:6379"
accept_content = ["json"]
task_serializer = "json"
result_serializer = "json"
redis_max_connections = 5
beat_schedule = {
    "reddit_memes_scrapper": {
        "task": "Reddit",
        "schedule": crontab(minute=0, hour="*"),
    },
}

FLASK_ENV = get_docker_secret(
    "FLASK_ENV", default=config("FLASK_ENV"), autocast_name=False
)

SQLALCHEMY_TRACK_MODIFICATIONS = False
user = get_docker_secret(
    "POSTGRES_USER", default=config("POSTGRES_USER"), autocast_name=False
)
password = get_docker_secret(
    "POSTGRES_PASSWORD", default=config("POSTGRES_PASSWORD"), autocast_name=False
)
db = get_docker_secret(
    "POSTGRES_DB", default=config("POSTGRES_DB"), autocast_name=False
)
SQLALCHEMY_DATABASE_URI = f"postgresql://{user}:{password}@sitedata:5432/{db}"

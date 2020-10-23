from celery import Celery
from celery_singleton import Singleton
from decouple import config
from flask import Flask

from controller.generated.models import db

CELERY_TASK_LIST = ["controller.tasks"]


def create_celery_app(app=None):
    app = app or create_app()
    celery = Celery(
        app.import_name,
        broker=app.config["CELERY_BROKER_URL"],
        include=CELERY_TASK_LIST,
    )
    celery.conf.update(app.config)

    class ContextTask(Singleton):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return Singleton.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery


def create_app(settings_override=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object("config.flask")
    db.init_app(app)
    return app

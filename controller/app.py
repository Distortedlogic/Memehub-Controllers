from celery import Celery
from celery_singleton import Singleton
from decouple import config
from flask import Flask
from flask_debugtoolbar import DebugToolbarExtension

from controller.generated.models import db

debug_toolbar = DebugToolbarExtension()


def create_celery_app(app=None):
    app = app or create_app()
    celery = Celery(
        app.import_name,
        broker=app.config["CELERY_BROKER_URL"],
        include=["controller.tasks"],
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
    debug_toolbar.init_app(app)
    print("app created")
    return app

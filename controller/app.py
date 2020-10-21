from controller.rai import load_models_to_redisai
from controller.redis.reddit import RedditReDB
from flask import Flask, jsonify, render_template, Blueprint, redirect, url_for
from flask_migrate import Migrate
from celery import Celery
from decouple import config
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from controller.sentry import integrate_sentry
from controller.extensions import debug_toolbar, db

import redisai

CELERY_TASK_LIST = ["controller.tasks"]


def create_celery_app(app=None):
    app = app or create_app()
    celery = Celery(
        app.import_name,
        broker=app.config["CELERY_BROKER_URL"],
        include=CELERY_TASK_LIST,
    )
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery


def create_app(settings_override=None):
    app = Flask(__name__, instance_relative_config=True)
    Migrate(app, db)
    app.config.from_object("config.flask")
    extensions(app)

    @app.route("/")
    def index():
        return redirect(url_for("demo.home"))

    @app.route("/redb_struct")
    def redb_struct():
        redb = RedditReDB()
        redb.update()
        return jsonify(redb.current)

    @app.route("/debug_sentry")
    def trigger_error():
        division_by_zero = 1 / 0
        return jsonify("I dIvIdEd By ZeRo")

    return app


def sentry():
    integrate_sentry(FlaskIntegration)
    integrate_sentry(CeleryIntegration)
    integrate_sentry(RedisIntegration)
    integrate_sentry(SqlalchemyIntegration)


def extensions(app):
    debug_toolbar.init_app(app)
    from controller.reddit.schema import RedditMeme, RedditScore, Redditor
    from controller.stonks.schema import TrainData, Template, NotMeme

    db.init_app(app)

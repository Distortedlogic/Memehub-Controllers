from controller.stonks.imgflip_trainer import ImgflipTrainer
from controller.stonks.imgflip_train_data import ImgflipTrainData
from controller.stonks.imgflip_templates import ImgflipTemplates
import click
from controller import APP


@click.group()
def cli():
    """ Run Imgflip Related Scripts"""
    pass


@click.command()
def template_db():
    """ Expand Reddit Meme Database for a Subreddit into the Future """

    with APP.app_context():
        ImgflipTemplates().build_db()

    return None


@click.command()
def train_data():
    """ Expand Reddit Meme Database for a Subreddit into the Future """

    with APP.app_context():
        ImgflipTrainData().run()

    return None


@click.command()
def create_models():
    """ Expand Reddit Meme Database for a Subreddit into the Future """

    with APP.app_context():
        ImgflipTrainer().run()

    return None


cli.add_command(template_db)
cli.add_command(train_data)
cli.add_command(create_models)

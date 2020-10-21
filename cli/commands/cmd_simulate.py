import click
from controller import APP
from controller.simulator.simulator import Simulator


@click.group()
def cli():
    """ Run Simulator Related Scripts"""
    pass


@click.command()
def db():
    """ Expand Reddit Meme Database for a Subreddit into the Future """

    with APP.app_context():
        Simulator().dbseed()

    return None


cli.add_command(db)

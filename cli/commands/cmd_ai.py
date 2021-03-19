import click
from src import APP
from src.stonks.updater import update


@click.group()
def cli():
    """ Run Imgflip Related Scripts"""
    pass


@click.command()
def sitedata():
    """
    Load models into redisai
    """
    with APP.app_context():
        update()

    return None


cli.add_command(sitedata)

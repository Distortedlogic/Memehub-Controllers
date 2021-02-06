import click
from controller import APP
from controller.stonks.rai import load_base_to_redisai, load_stonks_to_redisai
from controller.stonks.updater import update


@click.group()
def cli():
    """ Run Imgflip Related Scripts"""
    pass


@click.command()
def load_base():
    """
    Load models into redisai
    """
    load_base_to_redisai()

    return None


@click.command()
def load_stonks():
    """
    Load models into redisai
    """
    load_stonks_to_redisai()

    return None


@click.command()
def sitedata():
    """
    Load models into redisai
    """
    with APP.app_context():
        update()

    return None


cli.add_command(load_base)
cli.add_command(load_stonks)
cli.add_command(sitedata)

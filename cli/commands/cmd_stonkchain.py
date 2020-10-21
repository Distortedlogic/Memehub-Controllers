import click

from controller import APP
from controller.stonks.stonk_chain import StonkChain


@click.command()
def cli():
    """
    Load models into redisai
    """
    with APP.app_context():
        StonkChain().run()

    return None

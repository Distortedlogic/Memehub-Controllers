import click
from controller.rai import load_models_to_redisai


@click.command()
def cli():
    """
    Load models into redisai
    """
    load_models_to_redisai()

    return None

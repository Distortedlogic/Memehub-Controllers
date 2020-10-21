from controller.rai import load_models_to_redisai
import subprocess, click, os


@click.command()
def cli():
    """
    Load models into redisai
    """
    load_models_to_redisai()

    return None

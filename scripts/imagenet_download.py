from controller.stonks.not_a_meme.imagenet.imagenet import ImageNet
from controller import APP

if __name__ == "__main__":
    with APP.app_context():
        ImageNet().run()

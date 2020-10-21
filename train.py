from controller.stonks.imgflip_trainer import ImgflipTrainer
from controller import APP

if __name__ == "__main__":
    with APP.app_context():
        ImgflipTrainer().run()

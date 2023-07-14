from src.TextsummerizeProject.components import ModelTrainer
from src.TextsummerizeProject.config.configuration import ConfigrationManager
from src.TextsummerizeProject.logging import logger


class ModelTrainerPipeline():
    def __init__(self):
        pass

    def main(self):
        config = ConfigrationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()
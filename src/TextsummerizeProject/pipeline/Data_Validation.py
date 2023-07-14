from src.TextsummerizeProject.components import Datavalidation
from src.TextsummerizeProject.config.configuration import ConfigrationManager
from src.TextsummerizeProject.logging import logger

class datavalidationpipeline():
    def __init__(self):
        pass

    def main(self):
        config = ConfigrationManager()
        datavalidation_config = config.get_data_validation_config()
        datavalidation = Datavalidation(config=datavalidation_config)
        datavalidation.validate()
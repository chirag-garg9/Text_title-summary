from src.TextsummerizeProject.components import Datatransformation
from src.TextsummerizeProject.config.configuration import ConfigrationManager
from src.TextsummerizeProject.logging  import logger

class datatransformationpipeline():
    def __init__(self):
        pass

    def main(self):
        config = ConfigrationManager()
        datatransformation_config = config.get_data_transformation_config()
        datatransformation = Datatransformation(config=datatransformation_config)
        datatransformation.convert()

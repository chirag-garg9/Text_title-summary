from src.TextsummerizeProject.components import DataIngestion
from src.TextsummerizeProject.config.configuration import ConfigrationManager
from src.TextsummerizeProject.logging  import logger

class dataingestionpipeline():
    def __init__(self):
        pass

    def main(self):
        config = ConfigrationManager()
        Data_ingestion_config = config.get_data_ingestion_config()
        dataingestion = DataIngestion(config=Data_ingestion_config)
        dataingestion.download_file()
        dataingestion.extract_file()

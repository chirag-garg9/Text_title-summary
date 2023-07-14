from src.TextsummerizeProject.components import ModelEvaluation
from src.TextsummerizeProject.config.configuration import ConfigrationManager
from src.TextsummerizeProject.logging import logger

class ModelEvaluationPipeline():
    def __init__(self):
        pass

    def main(self):
        config = ConfigrationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()
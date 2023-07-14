
from src.TextsummerizeProject.constants import *
from src.TextsummerizeProject.utils.common import read_yaml, Create_directory
from src.TextsummerizeProject.entities import DataIngestionconfig
from src.TextsummerizeProject.entities import Datavalidationconfig
from src.TextsummerizeProject.entities import Datatransformationconfig
from src.TextsummerizeProject.entities import ModelTrainerConfig
from src.TextsummerizeProject.entities import ModelEvaluationConfig


class ConfigrationManager:
    def __init__(
            self, 
            config_path = CONFIGPATH, 
            params_path = PARAMSPATH
            ):
        
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        print(self.config.artifacts_root)
        Create_directory([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionconfig:
        config = self.config.data_ingestion

        Create_directory([config.root])

        data_ingestion_config = DataIngestionconfig(
            root_dir = config.root,
            source_url = config.source_url,
            local_data_file = config.local_data_file,
            unzip_directory = config.unzip_directory,
        ) 
        
        return data_ingestion_config
    
    def get_data_validation_config(self) -> Datavalidationconfig:
        config = self.config.data_validation

        Create_directory([config.root])

        data_validation_config = Datavalidationconfig(
            root_dir = config.root,
            status_file=config.status_file,
            all_required_files=config.all_required_files
        ) 
        
        return data_validation_config
    
    def get_data_transformation_config(self) -> Datatransformationconfig:
        config = self.config.data_transformation

        Create_directory([config.root])

        data_transformation_config = Datatransformationconfig(
            root_dir = config.root,
            data_path=config.data_path,
            data_path_test=config.data_path_test,
            tokenizer_name=config.tokenizer_name
        ) 
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        Create_directory([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt = config.model_ckpt,
            num_train_epochs = params.num_train_epochs,
            warmup_steps = params.warmup_steps,
            per_device_train_batch_size = params.per_device_train_batch_size,
            weight_decay = params.weight_decay,
            logging_steps = params.logging_steps,
            evaluation_strategy = params.evaluation_strategy,
            eval_steps = params.evaluation_strategy,
            save_steps = params.save_steps,
            gradient_accumulation_steps = params.gradient_accumulation_steps
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        Create_directory([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path = config.model_path,
            tokenizer_path = config.tokenizer_path,
            metric_file_name = config.metric_file_name
           
        )

        return model_evaluation_config

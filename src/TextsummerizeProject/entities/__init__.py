from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionconfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_directory: Path

@dataclass(frozen=True)
class Datavalidationconfig:
    root_dir: Path
    status_file: str
    all_required_files: list


@dataclass(frozen=True)
class Datatransformationconfig:
    root_dir: Path
    data_path: Path
    data_path_test: Path
    tokenizer_name: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
"""Module for configuration parameters dataclass"""

from dataclasses import dataclass
import pathlib
from typing import List, Dict, Union
import yaml

PROJECT_PATH = pathlib.Path(__file__).parent.resolve()
CONFIG_FILE_PATH = PROJECT_PATH / "config.yaml"

@dataclass
class DataConfig:
    """Configuration parameters dataclass"""
    train_data_path: str
    scalers_path: str
    model_path: str
    model_columns: List[str]
    binary_columns: List[str]
    numerical_columns: List[str]
    target_column: str
    cheap_car_threshold: int
    overused_car_threshold: int
    n_cv_splits: int

@dataclass
class ModelParamsConfig:
    """Configoration parameters for models parameters dataclass"""

    lasso_params: Dict[str, Union[str, int, float]]
    random_forest_params: Dict[str, Union[str, int, float]]
    boosting_params: Dict[str, Union[str, int, float]]

@dataclass
class UberConfig():
    """Uber config consisting of other configs"""
    data_config: DataConfig
    model_params_config: ModelParamsConfig

def create_config() -> UberConfig:
    """Run validation on config values."""
    parsed_config = fetch_config_from_yaml()

    __config = UberConfig(data_config=DataConfig(**parsed_config['data_params']),
                          model_params_config=ModelParamsConfig(**parsed_config['model_params']))

    return __config

def find_config_file() -> pathlib.Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml() -> Dict:
    """Parse YAML containing the package configuration."""
    cfg_path = find_config_file()
    if cfg_path:
        with open(cfg_path, "r", encoding='utf8') as conf_file:
            parsed_config = yaml.safe_load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


config = create_config()

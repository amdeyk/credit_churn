# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

import bank_churn_model

# Project Directories
PACKAGE_ROOT = Path(bank_churn_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"

DATASET_DIR = PACKAGE_ROOT / "databases"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    pipeline_name: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    unused_fields: List[str]

    Customer_Age_var: str
    Gender_var: str
    Dependent_count_var: str
    Education_Level_var: str
    Marital_Status_var: str
    Income_Category_var: str
    Card_Category_var: str
    Months_on_book_var: str
    Total_Relationship_Count_var: str
    Months_Inactive_12_mon_var: str
    Contacts_Count_12_mon_var: str
    Credit_Limit_var: str
    Total_Revolving_Bal_var: str
    Avg_Open_To_Buy_var: str
    Total_Amt_Chng_Q4_Q1_var: str
    Total_Trans_Amt_var: str
    Total_Trans_Ct_var: str
    Total_Ct_Chng_Q4_Q1_var: str
    Avg_Utilization_Ratio_var: str

    test_size: float
    random_state: int

    # Random Forest parameters
    rf_n_estimators: int
    rf_max_depth: int

    # CatBoost parameters
    cat_iterations: int


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""

    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH

    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config

    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()

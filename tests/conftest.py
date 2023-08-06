import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from sklearn.model_selection import train_test_split

from bank_churn_model.config.core import config
from bank_churn_model.processing.data_manager import load_dataset


@pytest.fixture
def sample_input_data():
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],     # predictors
        data[config.model_config.target],       # target
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,   # set the random seed here for reproducibility
    )

    return X_test, y_test

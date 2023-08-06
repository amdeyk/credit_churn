import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from bank_churn_model.config.core import config
from bank_churn_model.processing.features import UnusedFieldsDropper, CategoryImputer, OutlierClipper

def test_unused_fields_dropper(sample_input_data):
    # Given
    dropper = UnusedFieldsDropper(unused_fields=config.model_config.unused_fields)
    assert 'unused_field' in sample_input_data[0].columns

    # When
    subject = dropper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert 'unused_field' not in subject.columns

def test_category_imputer(sample_input_data):
    # Given
    imputer = CategoryImputer(variables=config.model_config.categorical_features)
    assert sample_input_data[0][config.model_config.categorical_features[0]].isnull().any()

    # When
    subject = imputer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert not subject[config.model_config.categorical_features[0]].isnull().any()

def test_outlier_clipper(sample_input_data):
    # Given
    clipper = OutlierClipper(variables=config.model_config.numeric_features)
    upper_threshold = sample_input_data[0][config.model_config.numeric_features[0]].quantile(0.95)
    assert sample_input_data[0][config.model_config.numeric_features[0]].max() > upper_threshold

    # When
    subject = clipper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject[config.model_config.numeric_features[0]].max() <= upper_threshold

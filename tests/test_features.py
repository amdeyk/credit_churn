import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from bank_churn_model.config.core import config
from bank_churn_model.processing.features import UnusedFieldsDropper, CategoryImputer, OutlierClipper

def test_unused_fields_dropper(sample_input_data):
    # Given
    dropper = UnusedFieldsDropper(unused_fields=config.model_config.unused_fields)
    for field in config.model_config.unused_fields:
        assert field in sample_input_data[0].columns

    # When
    subject = dropper.fit_transform(sample_input_data[0])

    # Then
    for field in config.model_config.unused_fields:
        assert field not in subject.columns




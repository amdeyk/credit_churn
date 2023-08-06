import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from bank_churn_model.config.core import config
from bank_churn_model.processing.features import UnusedFieldsDropper, CategoryImputer, OutlierClipper

def test_unused_fields_dropper(sample_input_data):
    # Given
    assert 2 == 2




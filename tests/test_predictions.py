"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from bank_churn_model.predict import make_prediction

def test_make_prediction(sample_input_data):
    # Given
    expected_num_of_predictions = len(sample_input_data[0])

    # When
    result = make_prediction(input_data = sample_input_data[0])

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.int64)
    assert result.get("errors") is None
    assert len(predictions) == expected_num_of_predictions

    _predictions = list(predictions)
    y_true = sample_input_data[1]

    accuracy = accuracy_score(y_true, _predictions)
    f1 = f1_score(y_true, _predictions)

    assert accuracy > 0.8
    assert f1 > 0.7

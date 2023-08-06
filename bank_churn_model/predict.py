import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bank_churn_model import __version__ as _version
from bank_churn_model.config.core import config
from bank_churn_model.processing.data_manager import load_pipeline
from bank_churn_model.processing.data_manager import pre_pipeline_preparation
from bank_churn_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
churn_pipe = load_pipeline(file_name = pipeline_file_name)

def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    validated_data = validated_data.reindex(columns = config.model_config.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = churn_pipe.predict(validated_data)
        results = {"predictions": predictions, "version": _version, "errors": errors}
        print(results)

    return results

if __name__ == "__main__":

    data_in = {
        'Customer_Age': [45], 
        'Gender': ['M'],
        'Dependent_count': [3],
        'Education_Level': ['High School'],
        'Marital_Status': ['Married'],
        'Income_Category': ['$60K - $80K'],
        'Card_Category': ['Blue'],
        'Months_on_book': [39],
        'Total_Relationship_Count': [5],
        'Months_Inactive_12_mon': [1],
        'Contacts_Count_12_mon': [3],
        'Credit_Limit': [12691.0],
        'Total_Revolving_Bal': [777],
        'Avg_Open_To_Buy': [11914.0],
        'Total_Amt_Chng_Q4_Q1': [1.335],
        'Total_Trans_Amt': [1144],
        'Total_Trans_Ct': [42],
        'Total_Ct_Chng_Q4_Q1': [1.625],
        'Avg_Utilization_Ratio': [0.061]
    }

    make_prediction(input_data = data_in)

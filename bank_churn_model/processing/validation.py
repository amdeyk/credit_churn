from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ValidationError

from bank_churn_model.config.core import config
from bank_churn_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        MultipleDataInputs(
            inputs=validated_data.to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    CLIENTNUM: Optional[int]
    Attrition_Flag: Optional[str]
    Customer_Age: Optional[int]
    Gender: Optional[str]
    Dependent_count: Optional[int]
    Education_Level: Optional[str]
    Marital_Status: Optional[str]
    Income_Category: Optional[str]
    Card_Category: Optional[str]
    Months_on_book: Optional[int]
    Total_Relationship_Count: Optional[int]
    Months_Inactive_12_mon: Optional[int]
    Contacts_Count_12_mon: Optional[int]
    Credit_Limit: Optional[float]
    Total_Revolving_Bal: Optional[int]
    Avg_Open_To_Buy: Optional[float]
    Total_Amt_Chng_Q4_Q1: Optional[float]
    Total_Trans_Amt: Optional[int]
    Total_Trans_Ct: Optional[int]
    Total_Ct_Chng_Q4_Q1: Optional[float]
    Avg_Utilization_Ratio: Optional[float]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

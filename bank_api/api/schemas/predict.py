from typing import Any, List, Optional
from pydantic import BaseModel
from bank_churn_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[str]]  # Changed from int to List[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Customer_Age": 40,
                        "Gender": "M",
                        "Dependent_count": 3,
                        "Education_Level": "Graduate",
                        "Marital_Status": "Married",
                        "Income_Category": "$60K - $80K",
                        "Card_Category": "Blue",
                        "Months_on_book": 36,
                        "Total_Relationship_Count": 5,
                        "Months_Inactive_12_mon": 1,
                        "Contacts_Count_12_mon": 2,
                        "Credit_Limit": 2300,
                        "Total_Revolving_Bal": 123,
                        "Avg_Open_To_Buy": 345,
                        "Total_Amt_Chng_Q4_Q1": 1.2,
                        "Total_Trans_Amt": 4000,
                        "Total_Trans_Ct": 54,
                        "Total_Ct_Chng_Q4_Q1": 0.77,
                        "Avg_Utilization_Ratio": 0.34
                    }
                ]
            }
        }

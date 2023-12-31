import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from bank_churn_model import __version__ as model_version
from bank_churn_model.predict import make_prediction

from api import __version__, schemas
from api.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Credit, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Credit(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    Churn prediction with the bank_churn_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    # Convert the 'predictions' NumPy array to a list
    if 'predictions' in results:
        results['predictions'] = results['predictions'].tolist()

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results

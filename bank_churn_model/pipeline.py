import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from bank_churn_model.config.core import config
from bank_churn_model.processing.features import CategoryImputer, OutlierClipper, UnusedFieldsDropper



numeric_features = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 
                    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 
                    'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 
                    'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

categorical_features = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

churn_pipe = Pipeline([
    ('unused_fields_dropper', UnusedFieldsDropper(unused_fields=config.model_config.unused_fields)),
    ('category_imputer', CategoryImputer(variables=categorical_features)),
    ('outlier_clipper', OutlierClipper(variables=numeric_features)),
    ('preprocessor', preprocessor),
    ('model_rf', RandomForestClassifier(n_estimators=config.model_config.rf_n_estimators,
                                    random_state=config.model_config.random_state))

])

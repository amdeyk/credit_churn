import sys
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from bank_churn_model.config.core import config


class UnusedFieldsDropper(BaseEstimator, TransformerMixin):
    """Remove unnecessary fields from the DataFrame."""

    def __init__(self, unused_fields: list):
        self.unused_fields = unused_fields

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for field in self.unused_fields:
            if field in X.columns:
                X.drop(labels=field, axis=1, inplace=True)
        return X


class CategoryImputer(BaseEstimator, TransformerMixin):
    """Impute missing categories."""

    def __init__(self, variables: list):
        self.variables = variables

    def fit(self, X, y=None):
        # logic to determine the category to be used for imputation
        self.most_frequent_categories = {var: X[var].mode()[0] for var in self.variables}
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var].fillna(self.most_frequent_categories[var], inplace=True)
        return X



class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers from the DataFrame."""

    def __init__(self, variables: list):
        self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            # Perform your outlier clipping operation here
            # For example, assuming you want to clip values greater than a certain threshold
            upper_threshold = X[var].quantile(0.95)  # change this as per your requirements
            X[var] = np.where(X[var] > upper_threshold, upper_threshold, X[var])
        return X


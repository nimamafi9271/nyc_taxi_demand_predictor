
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
import lightgbm as lgb
import pandas as pd



def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    adds one new column with average rides over 7, 14, 21, and 28 days
    """
    X["average_rides_last_4_weeks"]= 0.25 * (
                X[f"rides_previous_{7*24}_hour"] +\
                X[f"rides_previous_{2*7*24}_hour"] +\
                X[f"rides_previous_{3*7*24}_hour"] +\
                X[f"rides_previous_{4*7*24}_hour"]
    )

    return X


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):

    """
    scikit-learn data transformation that adds two columns:
    - hour of the day
    - day of the week
    also drops the original pickup_hour column"""


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_= X.copy()

        # Generate numeric columns from datetime
        X_["hour"]= X_["pickup_hour"].dt.hour
        X_["day_of_week"]= X_["pickup_hour"].dt.dayofweek

        return X_.drop(columns=["pickup_hour"])

def get_pipeline(**hyperparams) -> Pipeline:

    # sklearn transform
    add_average_rides_last_4_weeks= FunctionTransformer(
    average_rides_last_4_weeks, validate= False)

    # sklearn transform
    add_temporal_features= TemporalFeatureEngineer()

    # sklearn pipeline
    return make_pipeline(
    add_average_rides_last_4_weeks,
    add_temporal_features,
    lgb.LGBMRegressor(**hyperparams)
)

import os
from dotenv import load_dotenv
from src.paths import PARENT_DIR

# load key-value pairs from .env file in parent directory
load_dotenv(PARENT_DIR / ".env")

HOPSWORKS_PROJECT_NAME= "taxi_demand_predictor"
try:
    HOPSWORKS_API_KEY= os.environ["HOPSWORKS_API_KEY"]
except KeyError:
    raise Exception("create a .env file in the project root containing HOPSWORKS_API_KEY= <your_hopsworks_api_key>")

FEATURE_GROUP_NAME= "time_series_hourly_feature_group"
FEATURE_GROUP_VERSION= 1

FEATURE_VIEW_NAME= "time_series_hourly_feature_view"
FEATURE_VIEW_VERSION= 1
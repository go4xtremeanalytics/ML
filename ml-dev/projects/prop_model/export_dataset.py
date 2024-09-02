# import all the required dependecnies and modules

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib
# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set_theme(style="whitegrid")
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import pickle
from google.cloud import bigquery
from google.oauth2 import service_account
# from google.cloud import bigquery_storage_v1
import string
import xgboost as xgb
import importlib
import utility as pp
import category_encoders as ce
import warnings
warnings.filterwarnings("ignore")


def create_client (gcp_cred_json, ):
    import os
    os.environ['gcp_service_account'] = gcp_cred_json

    # Creating a BQ client
    import json
    # service_account_info = json.loads('ascendant-epoch-432900-m8-e97fb3b60497.json')

    with open(gcp_cred_json, 'r') as f:
            service_account_info = json.load(f)
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )
    client = bigquery.Client(credentials=credentials)
    return client


project_id = "ascendant-epoch-432900-m8"
dataset_id = "prop_model"
table_id = "churn_train"

def export_dataset(gcp_cred_json, project_id, dataset_id, table_id, input_query):
      client = create_client(gcp_cred_json)
      
      # Run the query
      query_job = client.query(input_query)

      # Fetch the results as a pandas DataFrame
      df = query_job.to_dataframe()

      # Print the shape of the DataFrame
      print(f"DataFrame shape: {df.shape}")

      # Display the first few rows
    #   print(df.display())
    #   print(display(df.head()))

      return df
      




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


def lift_and_ks(X_test, y_test, best_features, y_test_proba, y_test_pred, y_pred, y_proba, raw_data_res, id):

    ## Build final data
    df_predict_actual                   = pd.DataFrame()
    df_predict_actual['predict']        = y_pred
    df_predict_actual['predict_prob']   = y_proba
    df_predict_actual['customerID']         = raw_data_res[id]

    df_predict_actual.head()
    test_df = pd.concat([X_test[best_features], y_test], axis = 1)
    test_df['Churn_probability'] = y_test_proba
    test_df['y_pred'] = y_test_pred
    importlib.reload(pp)
    ks_df = pp.ks(data=test_df,target="Churn", prob= "Churn_probability")

    return ks_df
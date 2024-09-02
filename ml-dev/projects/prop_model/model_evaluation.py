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


def model_evaluation(best_model_after_fs, best_features, X_train, y_train, X_test, y_test, X_FE, y):
    importlib.reload(pp)
    [metrics_summary, y_test_pred, y_test_proba, y_train_proba,y_proba,y_pred] = pp.model_evaluation (best_model_after_fs,X_train[best_features], y_train, X_train[best_features], y_train, X_test[best_features], y_test, X_FE[best_features], y)
    from sklearn import metrics
    #calculate AUC of model
    auc = metrics.roc_auc_score(y_test, y_test_proba)

    print(metrics_summary)
    print(auc)

    return metrics_summary, y_test_pred, y_test_proba, y_train_proba,y_proba,y_pred, auc
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


def model_training(ratio, best_features, X_train, y_train):
    importlib.reload(pp)
    start=time.time()
    ### get best hyper parameters
    best_model_after_fs = pp.grid_search(X_train[best_features], y_train, cv =  3, ratio = ratio)
    end=time.time()
    print ("time_spend: "+str(end-start))

    # filename = 'Outputs/telecom_churn_model_v1.sav'
    # pickle.dump(best_model_after_fs, open(filename,'wb'))

    return best_model_after_fs
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


def feature_engineering(X_PP, y):
    import importlib
    importlib.reload(pp)
    Pipe_FE = Pipeline([
                    # ('Num_Col_Covar', pp.Num_Col_Covar(X=X_PP,  y = y, outcome_field = 'incomplete_instal')),
                    ('OHE_Encoding', pp.OHE_Encoding(X=X_PP, model_type = 'Train')),# Removing correlated Numerical Features
                    # ('WOE_Encoding', pp.WOE_Encoding( y = y)), #Weight Of Evidence Encoding for all Cat Columns - may change this to PCA
                    # ('Vif_Feature_Select', pp.Vif_Feature_Select(VIF_threshold = 3)), # Variance Inflation Factor for Feature Selection
                    
                    ])


    # fit_transform()
    X_FE = Pipe_FE.fit_transform(X_PP, y)
    print(X_FE.shape)
    X_FE.head()

    return X_FE
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


def pre_processing(X, cat_col, num_col):
    Pipe_PP = Pipeline([
                    
                    # Demo Data
                    ('Imputation_demo_col', pp.SimpleImputer(features = cat_col, strategy='constant', missing_values=np.nan,fill_value= 'None')),
                    
                    # Comp and Sevices Col
                    ('Imputation_comp_col', pp.SimpleImputer(features = num_col, strategy='constant', missing_values=np.nan,fill_value= 0)),
#                     ('Imputation_comp_service_col', pp.SimpleImputer(features = comp_service_col, strategy='constant', missing_values=np.nan,fill_value= 'None')),
                    
#                     # other_col
#                     ('Imputation_other_cat_col', pp.SimpleImputer(features = other_cat_col, strategy='constant', missing_values=np.nan,fill_value= 'None')),
#                     ('Imputation_other_num_col', pp.SimpleImputer(features = other_num_col, strategy='constant', missing_values=np.nan,fill_value= 0)),
                    
                    
#                     # All Columns 
#                     ('RareCategory', pp.RareCategoryEncoder(features_CAT_ = rare_col, category_min_pct=0.001, category_max_count=10)), #Droping levels that has very low proportion
                    ('Constant1', pp.Remove_ConstantFeatures(unique_threshold=1, missing_threshold=0.00)), # 
                    # ('Num_Col_Covar', pp.Num_Col_Covar(X=X,  y = y, outcome_field = 'incomplete_flag'))
                  ])
    # fit_transform()
    X_PP = Pipe_PP.fit_transform(X)
    print(X_PP.shape)
    X_PP.head()

    return X_PP
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


def feature_selection(X_train, y_train,  ratio,  importance_threshold = 0.70):
    ### get important fetures
    start=time.time()
    ### get best hyper parameters
    best_model = pp.grid_search(X_train, y_train, ratio, cv =3)
    end=time.time()
    print ("time_spend: "+str(end-start))

    feature_importances_df = pd.DataFrame()
    feature_importances_df['feature'] = X_train.columns
    feature_importances_df['Importance'] = best_model.feature_importances_
    feature_importances_df  = feature_importances_df.sort_values('Importance', ascending=False)
    feature_importances_df['ranking_importance'] = np.arange(1, feature_importances_df.shape[0]+1, 1)

    plt.figure(figsize = (15,6))
    sns.lineplot(data = feature_importances_df, x = 'ranking_importance', y = 'Importance')
    plt.show()

    print(feature_importances_df)


    feature_importances_df['Importance_cumsum'] = feature_importances_df['Importance'].cumsum()

    # can set importance_threshold to appropriate percentage of cumulative Importance that you want to cutoff features at
    
    print(len(feature_importances_df[round(feature_importances_df['Importance_cumsum'], 2) <= importance_threshold]), 'features at or below threshold of', str(int(round(importance_threshold * 100, 0))) + '%')
    print(feature_importances_df[round(feature_importances_df['Importance_cumsum'], 2) <= importance_threshold].head(5))
    feature_importances_df[round(feature_importances_df['Importance_cumsum'], 2) <= importance_threshold].tail(5)


    best_nb = len(feature_importances_df[round(feature_importances_df['Importance_cumsum'], 2) <= importance_threshold]) 
    best_features = list(feature_importances_df[0:best_nb]['feature'])
    len(best_features)

    # best_features = list(X_train.columns)

    # filename = 'Outputs/best_features_v1.sav'
    # pickle.dump(best_features, open(filename,'wb'))

    return best_features
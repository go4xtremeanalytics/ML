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
import utility as ut
import category_encoders as ce
import warnings
warnings.filterwarnings("ignore")
import export_dataset as ed
import define_X_y as xy
import pre_processing as pp
import feature_engineering as fe
import feature_selection as fs
import model_training as mt
import model_evaluation as me
import lift_and_ks as ks



import streamlit as st
from streamlit_shap import st_shap
import shap

from sklearn.model_selection import train_test_split
import xgboost

import numpy as np
import pandas as pd



#  Getting data from gcp
gcp_cred_json = 'ascendant-epoch-432900-m8-e97fb3b60497.json'
project_id = "ascendant-epoch-432900-m8"
dataset_id = "prop_model"
table_id = "new_churn"
input_query = """
    SELECT
      *
    FROM
      `ascendant-epoch-432900-m8.prop_model.new_churn`
    
      
"""


df = ed.export_dataset(gcp_cred_json, project_id, dataset_id, table_id, input_query)


print(df.head())


not_use = ['customerID', 'Churn']
id = ['ID']
target = ['Churn']


# Getting X and y columns

importlib.reload(pp)
cat_col, num_col, X, y  = xy.X_y_columns(df, not_use, id, target)


# Pre-processing
X_PP = pp.pre_processing(X, cat_col, num_col)


# Feature engineering
X_FE = fe.feature_engineering(X_PP, y)



# Test-train split
X_train, X_test, y_train, y_test = train_test_split(X_FE, y, test_size=0.25, stratify = y, random_state=24)

print('X_train shape : ', X_train.shape)
print('y_train shape : ', y_train.shape)

print('X_test shape  : ', X_test.shape)
print('y_test shape  : ', y_test.shape)

print('Disconnects in y_test : ', y_test.sum())


# Converting all columns to float type, for compatibility with XGBoost
for i in X_train:
    X_train[i] = X_train[i].astype('float')
for i in X_test:
    X_test[i] = X_test[i].astype('float')



# Calculate the ratio of negative class to positive class
ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])




# Feature selection
importlib.reload(pp)
best_features = fs.feature_selection(X_train, y_train,  ratio, importance_threshold = 0.70)


ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])



# Model training
importlib.reload(pp)
best_model = mt.model_training(ratio, best_features, X_train, y_train)

metrics_summary, y_test_pred, y_test_proba, y_train_proba,y_proba,y_pred, auc = me.model_evaluation(best_model, best_features, X_train, y_train, X_test, y_test, X_FE, y)


print("auc: ", auc)



## Build final data
df_predict_actual                   = pd.DataFrame()
df_predict_actual['predict']        = y_pred
df_predict_actual['predict_prob']   = y_proba
df_predict_actual['customerID']         = df['ID']

df_predict_actual.head()


# Predicting the test data

df_fixed    = df_predict_actual.sort_values(by = 'predict_prob', ascending = False).reset_index()
sample_size = df_predict_actual.shape[0]

# Segment the data into Decile size of 1
decile_size = sample_size/100 
df_fixed['Rank'] = ((df_predict_actual.index//decile_size)*1+1).astype('int64')

# df_fixed2 = df_fixed[df_fixed['Rank'] <= 20][['chc_id', 'Rank']]
df_fixed2 = df_fixed[['customerID', 'Rank']]
df_fixed2.head()


test_df = pd.concat([X_test[best_features], y_test], axis = 1)
test_df['Churn_probability'] = y_test_proba
test_df['y_pred'] = y_test_pred


test_df.Churn.value_counts()


# KS and lift analysis
importlib.reload(pp)
ks_df = ut.ks(data=test_df,target="Churn", prob= "Churn_probability")
print("KS and lift analysis: ", ks_df)



# SHAP
import shap
explainer = shap.Explainer(best_model, X_test[best_features])
shap_values = explainer(X_test[best_features])

# SHAP summary plot
shap.summary_plot(shap_values, X_test[best_features], plot_type= 'bar')



# SHAP beeswarm plot
shap.plots.beeswarm(shap_values, max_display = 20)





# st.title("SHAP in Streamlit")

# # compute SHAP values
# # explainer = shap.Explainer(best_model, X_test[best_features])
# # shap_values = explainer(X_test[best_features])

# st_shap(shap.plots.waterfall(shap_values[0]), height=400,  width = 750)
# st_shap(shap.plots.beeswarm(shap_values), height=400, width = 750)

# explainer = shap.TreeExplainer(best_model)
# shap_values = explainer.shap_values(X_test[best_features])

# st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_test[best_features].iloc[0,:]), height=200, width=1000)
# # st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000,:], X_test[best_features].iloc[:1000,:]), height=400, width=1000)



st.title('Predicting customer churn')
st.subheader('Created by: Xtreme Analytics')
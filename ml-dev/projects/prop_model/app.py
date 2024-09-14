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

from google.oauth2 import service_account
from google.cloud import bigquery

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)



# Perform query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    # Convert to list of dicts. Required for st.cache_data to hash the return value.
    rows = [dict(row) for row in rows_raw]
    return rows



rows = run_query("SELECT * FROM`ascendant-epoch-432900-m8.prop_model.new_churn` ")


df = pd.DataFrame(rows)

# #  Getting data from gcp
# gcp_cred_json = 'ascendant-epoch-432900-m8-e97fb3b60497.json'
# project_id = "ascendant-epoch-432900-m8"
# dataset_id = "prop_model"
# table_id = "new_churn"
# input_query = """
#     SELECT
#       *
#     FROM
#       `ascendant-epoch-432900-m8.prop_model.new_churn`
    
      
# """




# df = ed.export_dataset(gcp_cred_json, project_id, dataset_id, table_id, input_query)

df = df.drop(columns = ['int64_field_0'])

print(df.head())

print(df.columns)

not_use = ['customerID', 'Churn']
id = ['ID']
target = ['Churn']


# Getting X and y columns

importlib.reload(pp)
cat_col, num_col, X, y  = xy.X_y_columns(df, not_use, id, target)
num_col.remove('ID')

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

print(best_features)


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


rank_df = test_df.copy()
rank_df = rank_df.rename(columns = {"y_pred": "Churn_prediction", "bucket": "prediction_probability_range"})
rank_df['customerID']         = df['ID']
# Step 1: Create percentile ranks (100 is highest, 1 is lowest)
rank_df['Rank'] = pd.qcut(rank_df['Churn_probability'], q=100, labels=False, duplicates='drop')


# KS and lift analysis
importlib.reload(pp)
ks_df = ut.ks(data=test_df,target="Churn", prob= "Churn_probability")
print("KS and lift analysis: ", ks_df)







st.title('Predicting customer churn')
st.subheader('Created by: Xtreme Analytics')


# st.write(dashboard)


import streamlit as st
import pandas as pd


import streamlit as st
from streamlit_shap import st_shap
import shap


st.write("""
Churn Prediction Model helps to identify the high risk customers who will be likely to churn in the coming days. This app is scheduled to refresh every day with the new updated list of active customers and their predicted probability results for churning
""")
# rank_df = pd.read_csv("rank_df.csv")

shape = rank_df.shape

st.write(f"""
## Rank Dataframe:
* The Rank dataset is the final prediction result, that has information regarding all the customers with their churn prediction and prediction probability. Also, this has all the customers ranked from 1 to 100 where 1 being low risk to churn and 100 beign high risk to churn 
* This ranking is done once a day/week/month based on the business use case
* The shape of the resultant rank table is: {shape}
""")


rank_filter = st.slider("Pick the rank range that you want the result", min_value=1, max_value=100, value=[1, 100], step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

print(rank_filter)

min_val, max_val = rank_filter

result_df = rank_df[(rank_df['Rank'] >= min_val) & (rank_df['Rank'] <= max_val) ]

st.write(f"""Total of {result_df.shape[0]} customers are there in the selected range""")

st.write(f"""### Sample
            """)




st.dataframe(result_df.head())


# Function to convert the DataFrame to a CSV file
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Convert DataFrame to CSV
csv = convert_df_to_csv(result_df)

# Create a download button
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="result_df.csv",
    mime="text/csv",
)


st.write(f"""### Model Evaluation

            SHAP Waterfall plot for important features
            """)





# SHAP
import shap
explainer = shap.Explainer(best_model, X_test[best_features])
shap_values = explainer(X_test[best_features])


# Display the plot
st_shap(shap.plots.waterfall(shap_values[0]))



st.markdown("For further deatiled explaination, please check out the [explainer dashboard](http://192.168.1.98:9050/)") 

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.dashboards import ShapDependenceComponent, ShapSummaryComponent
explainer = ClassifierExplainer(best_model, X_test[best_features], y_test)
# explainer = shap.Explainer(best_model, X_test[best_features])
# shap_values = explainer(X_test[best_features])

dashboard = ExplainerDashboard(explainer,  
    importances=True,
    model_summary=True,
    contributions=True,
    whatif=True,
    shap_dependence=False,
    shap_interaction=True,
    shap_summary=True,
    decision_trees=False,
    hide_pdp=True,
    hide_whatifpdp=True
    )

dashboard.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)
# dashboard.run(use_waitress=True)  # or any other available port


# Action Items

# 1. Remove Partial Dependence Plot
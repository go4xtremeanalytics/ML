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


not_use = ['customerID', 'Churn']
id = ['customerID']
target = ['Churn']


def X_y_columns(df, not_use, id, target):
    raw_data_res = df.copy()

    print('Total customers: ', raw_data_res.shape[0])
    print('churners: ',raw_data_res['Churn'].sum())
    print('churn rate rate %: ', round(raw_data_res["Churn"].value_counts(normalize = True)[1] * 100,2))


    import pandas as pd

    # # Assuming 'df' is your DataFrame and 'column_name' is the problematic column
    # raw_data_res['TotalCharges'] = pd.to_numeric(raw_data_res['TotalCharges'], errors='coerce')
    # import numpy as np

    # raw_data_res['TotalCharges'] = raw_data_res['TotalCharges'].replace('', np.nan)
    # raw_data_res['TotalCharges'] = raw_data_res['TotalCharges'].astype(float)

    # Defining cat and num columns
    cat_col = []
    num_col = []

    for col in raw_data_res.columns:
        if raw_data_res[col].dtypes == 'object':
            cat_col.append(col)
        elif col not in target:
            num_col.append(col)

    print("total cat cols: ", len(cat_col))
    print("total num cols: ", len(num_col))

    
    print("Cat Columns descriptions")
    print("-" * 50)

    for col in cat_col:
        if (col not in not_use) and (col not in target):
            unique_values = raw_data_res[col].unique()
            print(f"Column: {col}")
            print(f"Number of unique values: {len(unique_values)}")
            print(f"Unique values: {unique_values}")
            print(f"Value counts:\n{raw_data_res[col].value_counts()}")
            print(f"Value counts normalized:\n{raw_data_res[col].value_counts(normalize = True)}")
            print("-" * 50)  # Separator for readability


    all_col = cat_col + num_col

    print("Length of all columns: ", len(all_col))
    raw_data_res_v1 = raw_data_res[all_col]
    raw_data_res_v1.shape


    # Defining independent and target features
    X = raw_data_res_v1.copy()
    y = raw_data_res['Churn']


    return cat_col, num_col, X, y
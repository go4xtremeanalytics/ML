"""
Author: Shivahari Revathi Venkateswaran
Date:   11/20/2023

Title: Create Custom Transformers to Primarily Conduct Preprocessing and Feature Engineering 
        

Purpose:
NOTE:  
"""


from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
import pandas as pd
# pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set_theme(style="whitegrid")
from sklearn.model_selection import train_test_split
import time
import pickle
from google.cloud import bigquery
from google.oauth2 import service_account
# from google.cloud import bigquery_storage_v1
import string
# import xgboost as xgb
from sklearn.utils import resample

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import (StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler, RobustScaler,\
                                   Binarizer, KBinsDiscretizer, QuantileTransformer, PowerTransformer,\
                                   PolynomialFeatures, OneHotEncoder, OrdinalEncoder)
from sklearn.metrics import accuracy_score as acc, recall_score as recall, precision_score as precision, roc_auc_score as auc, f1_score as f1, roc_curve, confusion_matrix

# To Connect GCP

from google.cloud import bigquery
from google.oauth2 import service_account
# from google.cloud import bigquery_storage_v1
from sklearn.model_selection import GridSearchCV

class bqConnect:

    def __init__(self, cred_json, project_id):
        self.cred_json = cred_json
        self.project_id = project_id
        credentials = service_account.Credentials.from_service_account_file(self.cred_json)  # if you want to use raw json instead of a json file, use: service_account.Credentials.from_service_account_info(self.cred_json)
        self.client = bigquery.Client(credentials= credentials, project=self.project_id)

    def __repr__(self):
        return f'Connection({self.project_id}, {self.client})'
    
    def dry_run(self, query):
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

        query_job = self.client.query(
            (query),
            job_config=job_config,
        )  # Make an API request.
        # A dry run query completes immediately.
        return ("This query will process {} GB.".format(query_job.total_bytes_processed/1000000000))

    def fetch_data(self, query):
        return self.client.query(query).to_dataframe()
    
    def write_table(self, df: pd.DataFrame, table_id: str, schema: list, write_disposition: str):
        # WRITE_TRUNCATE: If the table already exists, BigQuery overwrites the data, removes the constraints and uses the schema from the load job.
        # WRITE_APPEND: If the table already exists, BigQuery appends the data to the table.
        # WRITE_EMPTY: If the table already exists and contains data, a 'duplicate' error is returned in the job result.
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition = write_disposition, 
        )
        job = self.client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )  # Make an API request.
        job.result()  # Wait for the job to complete.

# class bqConnect:

#     def __init__(self, cred_json, project_id):
#         self.cred_json = cred_json
#         self.project_id = project_id
#         credentials = service_account.Credentials.from_service_account_file(self.cred_json)
#         self.client = bigquery.Client(credentials= credentials, project=self.project_id)

#     def dry_run(self, query):
#         job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False) 

#         query_job = self.client.query(
#             (query),
#             job_config=job_config,
#         )  # Make an API request.
#         # A dry run query completes immediately.
#         return ("This query will process {} GB.".format(query_job.total_bytes_processed/1000000000))

#     def fetch_data(self, query):
#         return self.client.query(query).to_dataframe()

class Calls_Preprocessing:
    '''
    Purpose
    ---------------
    A custom utility function
    - (1) to prepare the calls column 
    
    
    Parameters
    ---------------
    week, month: Input week and month for data selection 
   
    

    Returns
    ---------------
    dataset: a final data mess for further analysis
    
    '''
    
    def __init__(self, calls_col):
        
        self.calls_col = calls_col
        
    def fit(self, x, y = None):
        #----------------------------------------------------------
        for col in self.calls_col:
            if 'ivr' not in col:
                print('Given column is not a IVR call')
                break
        print('*'*50 + '\nPre-Processing: Calls_Preprocessing\n' + '*'*50 )    
        print('\n ivr_repeat_1_prevDept ======================') 
        print('Before : ', x['ivr_repeat_1_prevDept'].unique()) 
        x.loc[(x['ivr_repeat_1_prevDept'].str.upper() ==  'UNKNOWN'), 'ivr_repeat_1_prevDept'] = 'None' 
        x.loc[(x['ivr_repeat_1_prevDept'].str.upper() !=  'CUSTOMER SERVICE REP') & (x['ivr_repeat_1_prevDept'].str.upper() !=  'TECHNICAL SERVICE REP') & (x['ivr_repeat_1_prevDept'].str.upper() !=  'NONE'), 'ivr_repeat_1_prevDept'] = 'ALL_OTHER'    
        print('After  : ', x['ivr_repeat_1_prevDept'].unique())


        print('\n ivr_repeat_2_prevDept ======================') 
        print('Before : ', x['ivr_repeat_2_prevDept'].unique()) 
        x.loc[(x['ivr_repeat_2_prevDept'].str.upper() ==  'UNKNOWN'), 'ivr_repeat_2_prevDept'] = 'None' 
        x.loc[(x['ivr_repeat_2_prevDept'].str.upper() !=  'CUSTOMER SERVICE REP') & (x['ivr_repeat_2_prevDept'].str.upper() !=  'TECHNICAL SERVICE REP') & (x['ivr_repeat_2_prevDept'].str.upper() !=  'NONE'), 'ivr_repeat_2_prevDept'] = 'ALL_OTHER'    
        print('After  : ', x['ivr_repeat_2_prevDept'].unique())

        print('\n ivr_repeat_3_prevDept ======================') 
        print('Before : ', x['ivr_repeat_3_prevDept'].unique()) 
        x.loc[(x['ivr_repeat_3_prevDept'].str.upper() ==  'UNKNOWN'), 'ivr_repeat_3_prevDept'] = 'None' 
        x.loc[(x['ivr_repeat_3_prevDept'].str.upper() !=  'CUSTOMER SERVICE REP') & (x['ivr_repeat_3_prevDept'].str.upper() !=  'TECHNICAL SERVICE REP') & (x['ivr_repeat_3_prevDept'].str.upper() !=  'NONE'), 'ivr_repeat_3_prevDept'] = 'ALL_OTHER'    
        print('After  : ', x['ivr_repeat_3_prevDept'].unique())
        
        return self
    
    def transform(self, x):
              
        return x

    

class Upgrades_Preprocessing:
    '''
    Purpose
    ---------------
    A custom utility function
    - (1) to prepare the calls column 
    
    
    Parameters
    ---------------
    week, month: Input week and month for data selection 
   
    

    Returns
    ---------------
    dataset: a final data mess for further analysis
    
    '''
    
    def __init__(self, upgrades_col):
        self.upgrades_col = upgrades_col
        
    def fit(self, x, y = None):
        print('*'*50 + '\nPre-Processing: Upgrades_Preprocessing\n' + '*'*50 )
        #----------------------------------------------------------
        
        #---------------------------------------------------------------   
        return self
    def transform(self, x):
        for col in self.upgrades_col:
            if 'ivr' not in col:
                print('Given column is not a IVR call')
                break
        
        print('\n speed_m0 ======================') 
        print('Before : ', x['speed_m0'].unique()) 
        x['speed_m0'] = x['speed_m0'].str.replace('OPTIMUM ','')
        x['speed_m0'] = x['speed_m0'].str.replace('BROADBAND INTERNET ','')
        x['speed_m0'] = x['speed_m0'].str.replace('None','0')
        x['speed_m0'] = x['speed_m0'].str.replace('OOL','25')
        x['speed_m0'] = x['speed_m0'].str.replace('GBPS','000')
        x['speed_m0'] = x['speed_m0'].str.split('/').str[0]
        x['speed_m0'] = x['speed_m0'].str.split('-').str[0]
        x['speed_m0'] = x['speed_m0'].astype('float')
        print('After  : ', x['speed_m0'].unique())


        print('speed_m1 ======================') 
        print('Before : ', x['speed_m1'].unique()) 
        x['speed_m1'] = x['speed_m1'].str.replace('OPTIMUM ','')
        x['speed_m1'] = x['speed_m1'].str.replace('BROADBAND INTERNET ','')
        x['speed_m1'] = x['speed_m1'].str.replace('None','0')
        x['speed_m1'] = x['speed_m1'].str.replace('OOL','25')
        x['speed_m1'] = x['speed_m1'].str.replace('GBPS','000')
        x['speed_m1'] = x['speed_m1'].str.split('/').str[0]
        x['speed_m1'] = x['speed_m1'].str.split('-').str[0]
        x['speed_m1'] = x['speed_m1'].astype('float')
        print('After  : ', x['speed_m1'].unique())


        print('speed_m2 ======================') 
        print('Before : ', x['speed_m2'].unique()) 
        x['speed_m2'] = x['speed_m2'].str.replace('OPTIMUM ','')
        x['speed_m2'] = x['speed_m2'].str.replace('BROADBAND INTERNET ','')
        x['speed_m2'] = x['speed_m2'].str.replace('None','0')
        x['speed_m2'] = x['speed_m2'].str.replace('OOL','25')
        x['speed_m2'] = x['speed_m2'].str.replace('GBPS','000')
        x['speed_m2'] = x['speed_m2'].str.split('/').str[0]
        x['speed_m2'] = x['speed_m2'].str.split('-').str[0]
        x['speed_m2'] = x['speed_m2'].astype('float')
        print('After  : ', x['speed_m2'].unique())


        print('speed_m3 ======================') 
        print('Before : ', x['speed_m3'].unique()) 
        x['speed_m3'] = x['speed_m3'].str.replace('OPTIMUM ','')
        x['speed_m3'] = x['speed_m3'].str.replace('BROADBAND INTERNET ','')
        x['speed_m3'] = x['speed_m3'].str.replace('None','0')
        x['speed_m3'] = x['speed_m3'].str.replace('OOL','25')
        x['speed_m3'] = x['speed_m3'].str.replace('GBPS','000')
        x['speed_m3'] = x['speed_m3'].str.split('/').str[0]
        x['speed_m3'] = x['speed_m3'].str.split('-').str[0]
        x['speed_m3'] = x['speed_m3'].astype('float')
        print('After  : ', x['speed_m3'].unique())


        print('\n speed_m4 ======================') 
        print('Before : ', x['speed_m4'].unique()) 
        x['speed_m4'] = x['speed_m4'].str.replace('OPTIMUM ','')
        x['speed_m4'] = x['speed_m4'].str.replace('BROADBAND INTERNET ','')
        x['speed_m4'] = x['speed_m4'].str.replace('None','0')
        x['speed_m4'] = x['speed_m4'].str.replace('OOL','25')
        x['speed_m4'] = x['speed_m4'].str.replace('GBPS','000')   
        x['speed_m4'] = x['speed_m4'].str.split('/').str[0]
        x['speed_m4'] = x['speed_m4'].str.split('-').str[0] 
        x['speed_m4'] = x['speed_m4'].astype('float')
        print('After  : ', x['speed_m4'].unique())


        x['speed_max_m1m4'] = x[["speed_m1", "speed_m2", "speed_m3", "speed_m4"]].max(axis=1)
        x['speed_change_m4m0'] = np.where(x['speed_m0'] == x['speed_max_m1m4'], 'same' , (np.where(x['speed_m0'] > x['speed_max_m1m4'], 'higher', 'lower')))
        x = x.drop(['speed_m1', 'speed_m2', 'speed_m3','speed_m4','speed_max_m1m4'],axis=1)


        x['speed_m0'] = x['speed_m0'].astype('string')
        x["speed_m0"] = x["speed_m0"].str[:-2]
        x.loc[(x['speed_m0'].str.upper() !=  '0') & (x['speed_m0'].str.upper() !=  '25') & (x['speed_m0'].str.upper() !=  '30') & (x['speed_m0'].str.upper() !=  '50') & (x['speed_m0'].str.upper() !=  '100') & (x['speed_m0'].str.upper() !=  '200') & (x['speed_m0'].str.upper() !=  '300') & (x['speed_m0'].str.upper() !=  '400') & (x['speed_m0'].str.upper() !=  '500') & (x['speed_m0'].str.upper() !=  '1000'), 'speed_m0'] = 'ALL_OTHER'
        x['speed_m0'] = x['speed_m0'].astype('object')
        print("dropped speed_m1 to speed_m4 and created speed_change_m4m0")
        print(x['speed_m0'].unique())
        print(x['speed_change_m4m0'].unique())

        #--------------------------------------------------------------
        print('\n max_svod_m1 to max_svod_m3')
        x["max_svod_m1"] = np.where((x["max_svod_m1"].str.upper() == "Y"), 1, 0)
        x["max_svod_m2"] = np.where((x["max_svod_m2"].str.upper() == "Y"), 1, 0)
        x["max_svod_m3"] = np.where((x["max_svod_m3"].str.upper() == "Y"), 1, 0)
        x['max_svod_m1'] = x['max_svod_m1'].astype('int32')
        x['max_svod_m2'] = x['max_svod_m2'].astype('int32')
        x['max_svod_m3'] = x['max_svod_m3'].astype('int32')

        x['max_svod_max_m2m3'] = x[["max_svod_m2", "max_svod_m3"]].max(axis=1)
        x['max_svod_change_m3m1'] = np.where(x['max_svod_m1'] == x['max_svod_max_m2m3'], 'same' , (np.where(x['max_svod_m1'] > x['max_svod_max_m2m3'], 'up', 'down')))
        x = x.drop(['max_svod_m1', 'max_svod_m2', 'max_svod_m3', 'max_svod_max_m2m3'],axis=1)
        print(x['max_svod_change_m3m1'].unique())
        print('Dropped max_svod_m1 to max_svod_m3 and created max_svod_change_m3m1')

        #----------------------------------------------------------------
        print('\n hbo_svod_new_m1 to hbo_svod_new_m3')
        x["hbo_svod_new_m1"] = np.where((x["hbo_svod_new_m1"].str.upper() == "Y"), 1, 0)
        x["hbo_svod_new_m2"] = np.where((x["hbo_svod_new_m2"].str.upper() == "Y"), 1, 0)
        x["hbo_svod_new_m3"] = np.where((x["hbo_svod_new_m3"].str.upper() == "Y"), 1, 0)
        x['hbo_svod_new_m1'] = x['hbo_svod_new_m1'].astype('int32')
        x['hbo_svod_new_m2'] = x['hbo_svod_new_m2'].astype('int32')
        x['hbo_svod_new_m3'] = x['hbo_svod_new_m3'].astype('int32')

        x['hbo_svod_max_m2m3'] = x[["hbo_svod_new_m2", "hbo_svod_new_m3"]].max(axis=1)
        x['hbo_svod_change_m3m1'] = np.where(x['hbo_svod_new_m1'] == x['hbo_svod_max_m2m3'], 'same' , (np.where(x['hbo_svod_new_m1'] > x['hbo_svod_max_m2m3'], 'up', 'down')))
        x = x.drop(['hbo_svod_new_m1', 'hbo_svod_new_m2', 'hbo_svod_new_m3', 'hbo_svod_max_m2m3'],axis=1)
        print(x['hbo_svod_change_m3m1'].unique())
        print('Dropped hbo_svod_new_m1 to hbo_svod_new_m3 and created hbo_svod_change_m3m1')

        #----------------------------------------------------------------
        print('\n stz_enc_svod_m1 to stz_enc_svod_m3')
        x["stz_enc_svod_m1"] = np.where((x["stz_enc_svod_m1"].str.upper() == "Y"), 1, 0)
        x["stz_enc_svod_m2"] = np.where((x["stz_enc_svod_m2"].str.upper() == "Y"), 1, 0)
        x["stz_enc_svod_m3"] = np.where((x["stz_enc_svod_m3"].str.upper() == "Y"), 1, 0)
        x['stz_enc_svod_m1'] = x['stz_enc_svod_m1'].astype('int32')
        x['stz_enc_svod_m2'] = x['stz_enc_svod_m2'].astype('int32')
        x['stz_enc_svod_m3'] = x['stz_enc_svod_m3'].astype('int32')

        x['stz_enc_svod_max_m2m3'] = x[["stz_enc_svod_m2", "stz_enc_svod_m3"]].max(axis=1)
        x['stz_enc_svod_change_m3m1'] = np.where(x['stz_enc_svod_m1'] == x['stz_enc_svod_max_m2m3'], 'same' , (np.where(x['stz_enc_svod_m1'] > x['stz_enc_svod_max_m2m3'], 'up', 'down')))
        x = x.drop(['stz_enc_svod_m1', 'stz_enc_svod_m2', 'stz_enc_svod_m3', 'stz_enc_svod_max_m2m3'],axis=1)
        print(x['stz_enc_svod_change_m3m1'].unique())
        print('Dropped stz_enc_svod_m1 to stz_enc_svod_m3 and created stz_enc_svod_change_m3m1')

        #----------------------------------------------------------------

        #---------------------------------------------------------------
        print('\n curr_video_tier_desc_m0 ======================') 
        print('Before : ', x['curr_video_tier_desc_m0'].unique()) 
        x.loc[(x['curr_video_tier_desc_m0'].str.contains("basic", case=False, na = False)), 'curr_video_tier_desc_m0'] = 'BASIC'   
        x.loc[(x['curr_video_tier_desc_m0'].str.contains("economy", case=False, na = False)), 'curr_video_tier_desc_m0'] = 'ECONOMY' 
        x.loc[(x['curr_video_tier_desc_m0'].str.contains("core", case=False, na = False)), 'curr_video_tier_desc_m0'] = 'CORE'   
        x.loc[(x['curr_video_tier_desc_m0'].str.contains("value", case=False, na = False)), 'curr_video_tier_desc_m0'] = 'VALUE' 
        x.loc[(x['curr_video_tier_desc_m0'].str.contains("select", case=False, na = False)), 'curr_video_tier_desc_m0'] = 'SELECT'   
        x.loc[(x['curr_video_tier_desc_m0'].str.contains("premier", case=False, na = False)), 'curr_video_tier_desc_m0'] = 'PREMIER' 
        x.loc[(x['curr_video_tier_desc_m0'].str.contains("free", case=False, na = False)), 'curr_video_tier_desc_m0'] = 'FREE'   
        x.loc[(x['curr_video_tier_desc_m0'].str.contains("bulk", case=False, na = False)), 'curr_video_tier_desc_m0'] = 'BULK'  
        x.loc[(x['curr_video_tier_desc_m0'].str.contains("family", case=False, na = False)), 'curr_video_tier_desc_m0'] = 'FAMILY' 
        x.loc[(x['curr_video_tier_desc_m0'].str.upper() !=  'BASIC') & (x['curr_video_tier_desc_m0'].str.upper() !=  'ECONOMY') & (x['curr_video_tier_desc_m0'].str.upper() !=  'CORE') \
              & (x['curr_video_tier_desc_m0'].str.upper() !=  'VALUE') & (x['curr_video_tier_desc_m0'].str.upper() !=  'SELECT') & (x['curr_video_tier_desc_m0'].str.upper() !=  'PREMIER') \
              # & (x['curr_video_tier_desc_m0'].str.upper() !=  'BULK') & (x['curr_video_tier_desc_m0'].str.upper() !=  'FREE') & (x['curr_video_tier_desc_m0'].str.upper() !=  'FAMILY') \
              & (x['curr_video_tier_desc_m0'].str.upper() !=  'NONE'), 'curr_video_tier_desc_m0'] = 'ALL_OTHER'
        print('After  : ', x['curr_video_tier_desc_m0'].unique())

        print('curr_video_tier_desc_m1 ======================') 
        print('Before : ', x['curr_video_tier_desc_m1'].unique()) 
        x.loc[(x['curr_video_tier_desc_m1'].str.contains("basic", case=False, na = False)), 'curr_video_tier_desc_m1'] = 'BASIC'   
        x.loc[(x['curr_video_tier_desc_m1'].str.contains("economy", case=False, na = False)), 'curr_video_tier_desc_m1'] = 'ECONOMY' 
        x.loc[(x['curr_video_tier_desc_m1'].str.contains("core", case=False, na = False)), 'curr_video_tier_desc_m1'] = 'CORE'   
        x.loc[(x['curr_video_tier_desc_m1'].str.contains("value", case=False, na = False)), 'curr_video_tier_desc_m1'] = 'VALUE' 
        x.loc[(x['curr_video_tier_desc_m1'].str.contains("select", case=False, na = False)), 'curr_video_tier_desc_m1'] = 'SELECT'   
        x.loc[(x['curr_video_tier_desc_m1'].str.contains("premier", case=False, na = False)), 'curr_video_tier_desc_m1'] = 'PREMIER' 
        x.loc[(x['curr_video_tier_desc_m1'].str.contains("free", case=False, na = False)), 'curr_video_tier_desc_m1'] = 'FREE'   
        x.loc[(x['curr_video_tier_desc_m1'].str.contains("bulk", case=False, na = False)), 'curr_video_tier_desc_m1'] = 'BULK'  
        x.loc[(x['curr_video_tier_desc_m1'].str.contains("family", case=False, na = False)), 'curr_video_tier_desc_m1'] = 'FAMILY' 
        x.loc[(x['curr_video_tier_desc_m1'].str.upper() !=  'BASIC') & (x['curr_video_tier_desc_m1'].str.upper() !=  'ECONOMY') & (x['curr_video_tier_desc_m1'].str.upper() !=  'CORE') \
              & (x['curr_video_tier_desc_m1'].str.upper() !=  'VALUE') & (x['curr_video_tier_desc_m1'].str.upper() !=  'SELECT') & (x['curr_video_tier_desc_m1'].str.upper() !=  'PREMIER') \
              # & (x['curr_video_tier_desc_m1'].str.upper() !=  'BULK') & (x['curr_video_tier_desc_m1'].str.upper() !=  'FREE') & (x['curr_video_tier_desc_m1'].str.upper() !=  'FAMILY') \
                             & (x['curr_video_tier_desc_m1'].str.upper() !=  'NONE'), 'curr_video_tier_desc_m1'] = 'ALL_OTHER'
        print('After  : ', x['curr_video_tier_desc_m1'].unique())


        print('curr_video_tier_desc_m2 ======================') 
        print('Before : ', x['curr_video_tier_desc_m2'].unique()) 
        x.loc[(x['curr_video_tier_desc_m2'].str.contains("basic", case=False, na = False)), 'curr_video_tier_desc_m2'] = 'BASIC'   
        x.loc[(x['curr_video_tier_desc_m2'].str.contains("economy", case=False, na = False)), 'curr_video_tier_desc_m2'] = 'ECONOMY' 
        x.loc[(x['curr_video_tier_desc_m2'].str.contains("core", case=False, na = False)), 'curr_video_tier_desc_m2'] = 'CORE'   
        x.loc[(x['curr_video_tier_desc_m2'].str.contains("value", case=False, na = False)), 'curr_video_tier_desc_m2'] = 'VALUE' 
        x.loc[(x['curr_video_tier_desc_m2'].str.contains("select", case=False, na = False)), 'curr_video_tier_desc_m2'] = 'SELECT'   
        x.loc[(x['curr_video_tier_desc_m2'].str.contains("premier", case=False, na = False)), 'curr_video_tier_desc_m2'] = 'PREMIER' 
        x.loc[(x['curr_video_tier_desc_m2'].str.contains("free", case=False, na = False)), 'curr_video_tier_desc_m2'] = 'FREE'   
        x.loc[(x['curr_video_tier_desc_m2'].str.contains("bulk", case=False, na = False)), 'curr_video_tier_desc_m2'] = 'BULK'  
        x.loc[(x['curr_video_tier_desc_m2'].str.contains("family", case=False, na = False)), 'curr_video_tier_desc_m2'] = 'FAMILY' 
        x.loc[(x['curr_video_tier_desc_m2'].str.upper() !=  'BASIC') & (x['curr_video_tier_desc_m2'].str.upper() !=  'ECONOMY') & (x['curr_video_tier_desc_m2'].str.upper() !=  'CORE') \
              & (x['curr_video_tier_desc_m2'].str.upper() !=  'VALUE') & (x['curr_video_tier_desc_m2'].str.upper() !=  'SELECT') & (x['curr_video_tier_desc_m2'].str.upper() !=  'PREMIER') \
              # & (x['curr_video_tier_desc_m2'].str.upper() !=  'BULK') & (x['curr_video_tier_desc_m2'].str.upper() !=  'FREE') & (x['curr_video_tier_desc_m2'].str.upper() !=  'FAMILY') \
                             & (x['curr_video_tier_desc_m2'].str.upper() !=  'NONE'), 'curr_video_tier_desc_m2'] = 'ALL_OTHER'
        print('After  : ', x['curr_video_tier_desc_m2'].unique())


        print('curr_video_tier_desc_m3 ======================') 
        print('Before : ', x['curr_video_tier_desc_m3'].unique()) 
        x.loc[(x['curr_video_tier_desc_m3'].str.contains("basic", case=False, na = False)), 'curr_video_tier_desc_m3'] = 'BASIC'   
        x.loc[(x['curr_video_tier_desc_m3'].str.contains("economy", case=False, na = False)), 'curr_video_tier_desc_m3'] = 'ECONOMY' 
        x.loc[(x['curr_video_tier_desc_m3'].str.contains("core", case=False, na = False)), 'curr_video_tier_desc_m3'] = 'CORE'   
        x.loc[(x['curr_video_tier_desc_m3'].str.contains("value", case=False, na = False)), 'curr_video_tier_desc_m3'] = 'VALUE' 
        x.loc[(x['curr_video_tier_desc_m3'].str.contains("select", case=False, na = False)), 'curr_video_tier_desc_m3'] = 'SELECT'   
        x.loc[(x['curr_video_tier_desc_m3'].str.contains("premier", case=False, na = False)), 'curr_video_tier_desc_m3'] = 'PREMIER' 
        x.loc[(x['curr_video_tier_desc_m3'].str.contains("free", case=False, na = False)), 'curr_video_tier_desc_m3'] = 'FREE'   
        x.loc[(x['curr_video_tier_desc_m3'].str.contains("bulk", case=False, na = False)), 'curr_video_tier_desc_m3'] = 'BULK'  
        x.loc[(x['curr_video_tier_desc_m3'].str.contains("family", case=False, na = False)), 'curr_video_tier_desc_m3'] = 'FAMILY' 
        x.loc[(x['curr_video_tier_desc_m3'].str.upper() !=  'BASIC') & (x['curr_video_tier_desc_m3'].str.upper() !=  'ECONOMY') & (x['curr_video_tier_desc_m3'].str.upper() !=  'CORE') \
              & (x['curr_video_tier_desc_m3'].str.upper() !=  'VALUE') & (x['curr_video_tier_desc_m3'].str.upper() !=  'SELECT') & (x['curr_video_tier_desc_m3'].str.upper() !=  'PREMIER') \
              # & (x['curr_video_tier_desc_m3'].str.upper() !=  'BULK') & (x['curr_video_tier_desc_m3'].str.upper() !=  'FREE') & (x['curr_video_tier_desc_m3'].str.upper() !=  'FAMILY') \
                             & (x['curr_video_tier_desc_m3'].str.upper() !=  'NONE'), 'curr_video_tier_desc_m3'] = 'ALL_OTHER'
        print('After  : ', x['curr_video_tier_desc_m3'].unique())


        print('curr_video_tier_desc_m4 ======================') 
        print('Before : ', x['curr_video_tier_desc_m4'].unique()) 
        x.loc[(x['curr_video_tier_desc_m4'].str.contains("basic", case=False, na = False)), 'curr_video_tier_desc_m4'] = 'BASIC'   
        x.loc[(x['curr_video_tier_desc_m4'].str.contains("economy", case=False, na = False)), 'curr_video_tier_desc_m4'] = 'ECONOMY' 
        x.loc[(x['curr_video_tier_desc_m4'].str.contains("core", case=False, na = False)), 'curr_video_tier_desc_m4'] = 'CORE'   
        x.loc[(x['curr_video_tier_desc_m4'].str.contains("value", case=False, na = False)), 'curr_video_tier_desc_m4'] = 'VALUE' 
        x.loc[(x['curr_video_tier_desc_m4'].str.contains("select", case=False, na = False)), 'curr_video_tier_desc_m4'] = 'SELECT'   
        x.loc[(x['curr_video_tier_desc_m4'].str.contains("premier", case=False, na = False)), 'curr_video_tier_desc_m4'] = 'PREMIER' 
        x.loc[(x['curr_video_tier_desc_m4'].str.contains("free", case=False, na = False)), 'curr_video_tier_desc_m4'] = 'FREE'   
        x.loc[(x['curr_video_tier_desc_m4'].str.contains("bulk", case=False, na = False)), 'curr_video_tier_desc_m4'] = 'BULK'  
        x.loc[(x['curr_video_tier_desc_m4'].str.contains("family", case=False, na = False)), 'curr_video_tier_desc_m4'] = 'FAMILY' 
        x.loc[(x['curr_video_tier_desc_m4'].str.upper() !=  'BASIC') & (x['curr_video_tier_desc_m4'].str.upper() !=  'ECONOMY') & (x['curr_video_tier_desc_m4'].str.upper() !=  'CORE') \
              & (x['curr_video_tier_desc_m4'].str.upper() !=  'VALUE') & (x['curr_video_tier_desc_m4'].str.upper() !=  'SELECT') & (x['curr_video_tier_desc_m4'].str.upper() !=  'PREMIER') \
              # & (x['curr_video_tier_desc_m4'].str.upper() !=  'BULK') & (x['curr_video_tier_desc_m4'].str.upper() !=  'FREE') & (x['curr_video_tier_desc_m4'].str.upper() !=  'FAMILY') \
                             & (x['curr_video_tier_desc_m4'].str.upper() !=  'NONE'), 'curr_video_tier_desc_m4'] = 'ALL_OTHER'
        print('After  : ', x['curr_video_tier_desc_m4'].unique())


        x = x.replace({'curr_video_tier_desc_m0':{'None' : 0, 'BASIC' : 1, 'ECONOMY' : 2, 'CORE' : 3, 'VALUE' : 4, 'SELECT' : 5, 'PREMIER' : 6, 'ALL_OTHER': -1},
                   'curr_video_tier_desc_m1':{'None' : 0, 'BASIC' : 1, 'ECONOMY' : 2, 'CORE' : 3, 'VALUE' : 4, 'SELECT' : 5, 'PREMIER' : 6, 'ALL_OTHER': -1 },
                   'curr_video_tier_desc_m2':{'None' : 0, 'BASIC' : 1, 'ECONOMY' : 2, 'CORE' : 3, 'VALUE' : 4, 'SELECT' : 5, 'PREMIER' : 6, 'ALL_OTHER': -1 },
                   'curr_video_tier_desc_m3':{'None' : 0, 'BASIC' : 1, 'ECONOMY' : 2, 'CORE' : 3, 'VALUE' : 4, 'SELECT' : 5, 'PREMIER' : 6, 'ALL_OTHER': -1 },
                   'curr_video_tier_desc_m4':{'None' : 0, 'BASIC' : 1, 'ECONOMY' : 2, 'CORE' : 3, 'VALUE' : 4, 'SELECT' : 5, 'PREMIER' : 6, 'ALL_OTHER': -1 }})
        print('After encoding : ', x['curr_video_tier_desc_m0'].unique())


        x['video_tier_max_m1m4'] = x[["curr_video_tier_desc_m1", "curr_video_tier_desc_m2", "curr_video_tier_desc_m3", "curr_video_tier_desc_m4"]].max(axis=1)
        x['video_tier_change_m4m0'] = np.where(x['curr_video_tier_desc_m0'] == x['video_tier_max_m1m4'], 'same' , (np.where(x['curr_video_tier_desc_m0'] > x['video_tier_max_m1m4'], 'higher', 'lower')))


        x.loc[(x['video_tier_change_m4m0'] != 'same') & (x['curr_video_tier_desc_m0'] == -1), 'video_tier_change_m4m0'] = 'OTHER' 
        x.loc[(x['video_tier_change_m4m0'] != 'same') & (x['video_tier_max_m1m4'] == -1), 'video_tier_change_m4m0'] = 'OTHER' 
        x = x.replace({'curr_video_tier_desc_m0':{0 : 'None', 1 : 'BASIC' , 2 :'ECONOMY' , 3 : 'CORE', 4 : 'VALUE',  5 : 'SELECT', 6 : 'PREMIER', -1 : 'ALL_OTHER'}})
        x = x.drop(['curr_video_tier_desc_m1', 'curr_video_tier_desc_m2', 'curr_video_tier_desc_m3','curr_video_tier_desc_m4','video_tier_max_m1m4'],axis=1)
        print('After decoding : ', x['curr_video_tier_desc_m0'].unique())
        print('Dropped curr_video_tier_desc_m1 to curr_video_tier_desc_m4 and created video_tier_change_m4m0 variable')
        #----------------------------------------------------------
        print('\n curr_ov_tier_desc_m0 to curr_ov_tier_desc_m0 ==================================')
        x['ov_tier_change_m4m0'] = np.where((x['curr_ov_tier_desc_m0'] == x['curr_ov_tier_desc_m1']) & 
                                          (x['curr_ov_tier_desc_m0'] == x['curr_ov_tier_desc_m2']) & 
                                          (x['curr_ov_tier_desc_m0'] == x['curr_ov_tier_desc_m3']) &
                                          (x['curr_ov_tier_desc_m0'] == x['curr_ov_tier_desc_m4']), 0 , 1)
        x = x.drop(['curr_ov_tier_desc_m0', 'curr_ov_tier_desc_m1', 'curr_ov_tier_desc_m2', 'curr_ov_tier_desc_m3','curr_ov_tier_desc_m4'],axis=1)
        print('created ov_tier_change_m4m0 variable')
        print("dropped curr_ov_tier_desc_m1 to curr_ov_tier_desc_m4")
        

              
        return x



    

class Treat_Outliers():
    
    '''
    Purpose
    ---------------
    A custom trasformer
    - (1) to treat the outliers 
    - (2) by replacing the values that are greater than 95th percentile by 95th percentile value or by a given value if the value is greater than the given value 
    
    Parameters
    ---------------
    features: list of features for the outlier treatment 
    value: value that is treated as an outlier, Default value is None
    percentile: percentile that is treated as an outlier, Default value is None

    Returns
    ---------------
    X: a df with the remaining features
    
    '''
    
    def __init__(self, features, ignore_none = False, upper_value = None, lower_value = None, upper_percentile = None, lower_percentile = None, Negative_value = None):
        
        self.features = features 
        self.ignore_none = ignore_none
        self.upper_value = upper_value
        self.lower_value = lower_value
        self.Negative_value = Negative_value
        self.upper_percentile = upper_percentile
        self.lower_percentile = lower_percentile
        
    
    def fit(self,X, y = None):
        print('*'*50 + '\nPre-Processing: Treat_Outliers\n' + '*'*50 )
            # replace outliers with 99th upper_percentile
        X[self.features] = X[self.features].astype('int')
        for col in self.features:
            if X[col].isnull().any() and self.ignore_none is False:
                raise Warning("Outliers are treated with Null values, Please remove the Null values")
        for col in self.features:
            if self.upper_value is not None:
                X[col] = X[col].apply(lambda x: upper_value if x > upper_value else x)
                print('*'*50 + "\nOutliers in the column '" + str(col) + "' are replaced by the given upper_value\n")
            if self.lower_value is not None:
                X[col] = X[col].apply(lambda x: lower_value if x < lower_value else x)
                print('*'*50 + "\nOutliers in the column '" + str(col) + "' are replaced by the given lower_value\n")
            if self.Negative_value is not None:
                X[col] = X[col].apply(lambda x: 0 if x < 0 else x)
                print('*'*50 + "\nNegative values in the column '" + str(col) + "'  are replaced by Zero\n")
            if self.upper_percentile is not None:
                upper_quantile = X[col].dropna().quantile(self.upper_percentile)
                X[col] = X[col].apply(lambda x: upper_quantile if x > upper_quantile else x)
                print('*'*50 + "\nOutliers in the column '" + str(col) + "'  are replaced by the given upper_quantile\n")
            if self.lower_percentile is not None:
                lower_quantile = X[col].dropna().quantile(self.lower_quantile)
                X[col] = X[col].apply(lambda x: lower_quantile if x < lower_quantile else x)
                print('*'*50 + "\nOutliers in the column '" + str(col) + "'  are replaced by the given lower_quantile\n")
      
            if self.upper_value is None and self.lower_value is None and self.Negative_value is None and self.upper_percentile is None and self.lower_percentile is None: 
                raise Exception("Please provide a value or a percentile for outlier treatment")
        return self
    
    def transform(self, X):
        
        #for col in self.features:           
            
            ## droping negative observation 
            #X = X[X[col]>=0]
        # X_temp = X.copy()
        
        return X



class SimpleImputer():
    
    '''
    Purpose
    ---------------
    A custom trasformer
    - (1) for imputation 
    - (2) by mean, median, mode or any constant values
    
    Parameters
    ---------------
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

    Returns
    ---------------
    X: a df with the remaining features
    
    '''
    
    def __init__(self, features, missing_values=np.nan, strategy = 'mean', fill_value=None,copy=True,add_indicator=False,keep_empty_features=False,):
        self.features = features 
        self.missing_values = missing_values 
        self.strategy = strategy
        self.fill_value = fill_value
        # self.verbose = verbose
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        
    def fit(self,X, y = None):
        print('*'*50 + '\nPre-Processing: SimpleImputer\n' + '*'*50 )
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=self.strategy, missing_values=self.missing_values,fill_value=self.fill_value, copy=self.copy,add_indicator=self.add_indicator,keep_empty_features=self.keep_empty_features,)
        imputer = imputer.fit(X[self.features])
        X[self.features] = imputer.transform(X[self.features])
        if self.strategy == 'constant' and self.fill_value == 0:
            X[self.features] = X[self.features].astype('int')
        return self
    def transform(self, X):
        return X

    

class Remove_ConstantFeatures(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer
    - (1) to identify features with a single unique value and 
    - (2) to remove those constant features
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    unique_threshold: number of unique values >= 1 in integer
       - default: unique_threshold=1
    missing_threshold: missing percentage ~ [0, 1]
       - default: missing_threshold=0.00     
           - missing_threshold=0.00 --> Focus on features with full non-missing values.
           - missing_threshold=1.00 --> Focus on all features regardless of missing pct.
    
    Returns
    ---------------
    datafrme
        - X: a df that consists of selected features after dropping via '.tranform(X)'        
        - summary_dropped_: a df that includes dropped features with simple summary statistics
        - summary_dropped_NUM_: a df that includes dropped NUM features with full summary statistics
        - summary_dropped_CAT_: a df that includes dropped CAT features with full summary statistics
    list
        - features_dropped_: features that drop due to non-unique values
        - features_kept_: features that are kept
    
    References
    ---------------        
    Feature Selector: https://github.com/WillKoehrsen/feature-selector
    '''
    
    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    features_irrelevant = ['bdate', 'bmonth2', 'churn', 'churn2', 'comparea', 'compdisco', 'complete2', 'count', \
                           'cust_tenure_start_date', 'cutdate', 'disco', 'disco_rsn', 'drform', 'drform2', 'dt', \
                           'edate', 'file', 'filedt', 'fiosclust', 'fiosind', 'ftax', 'gf_mig_date', 'install', \
                           'max_repeat_wfindate', 'max_repeat_wip_id', 'min_disco_wfindate', 'min_disco_wip_id', \
                           'ont_wfindate', 'prefios', 'prod_m0', 'prod_m1', 'prod_m2', 'prod_m3', 'prod_m4', \
                           'rc_str', 'restart', 'spindate', 'uverse', 'video_business_class', 'vidtengrp3', \
                           'wstat', 'zip5', 'vol_pend_disco', 'pending_drform', 'account_pend_disco_frq', \
                           'chc_id', 'house', 'status']
    
    
    def __init__(self, unique_threshold=1, missing_threshold=0.00):
        self.unique_threshold  = unique_threshold
        self.missing_threshold = missing_threshold
        
        
        

    def fit(self, X, y=None):
        print('*'*50 + '\nFeature Engineering: Remove_ConstantFeatures\n' + '*'*50 + \
              '\n- It will remove features with {} unique value(s).\n'.format(self.unique_threshold))
        # 0. Remove Irrelevant Features
        features_relevant = list(set(X.columns) - set(self.features_irrelevant))
        features_relevant.sort()
        features_NUM = X[features_relevant].select_dtypes(exclude=[object, 'category']).columns.tolist()
        features_CAT = X[features_relevant].select_dtypes(include=[object, 'category']).columns.tolist()

        # 1. Compute Unique Values of All Features   
        # Unique values should be considered in conjuection with missing pct
        missing_full_      = pd.DataFrame(X[features_relevant].isnull().sum() / X.shape[0]).reset_index().\
                             rename(columns = {'index': 'feature', 0: 'missing_pct'})
                             
        self.summary_full_ = pd.DataFrame(X[features_relevant].nunique()).reset_index().\
                             rename(columns = {'index': 'feature', 0: 'nunique'}).\
                             merge(missing_full_, left_on='feature', right_on='feature', how='inner').\
                             sort_values('nunique', ascending = False).\
                             reset_index(drop=True)
        
        # 2. Identify Features with a single unique value
        flag_dropped          = (self.summary_full_['nunique'] <= self.unique_threshold) & \
                                (self.summary_full_['missing_pct'] <= self.missing_threshold)
                                
        self.summary_dropped_ = self.summary_full_[flag_dropped].reset_index(drop=True)
        self.summary_kept_    = self.summary_full_[~flag_dropped].reset_index(drop=True)
        
        # 3. Make a List of Dropped Features
        self.features_kept_    = self.summary_kept_.sort_values('feature')['feature'].tolist()        
        self.features_dropped_ = self.summary_dropped_.sort_values('feature')['feature'].tolist()
        features_dropped_NUM   = [fe for fe in self.features_dropped_ if fe in features_NUM]
        features_dropped_CAT   = [fe for fe in self.features_dropped_ if fe in features_CAT]        

        # 4. Make a df of dropped features with summary statistics
        # Note: A custom utility function, 'Summarize_Features()' is used.
        
        if (len(self.features_dropped_)==0):
            self.summary_dropped_NUM_ = self.summary_dropped_.copy()
            self.summary_dropped_CAT_ = self.summary_dropped_.copy()
        elif (len(self.features_dropped_)>0) & (len(features_dropped_NUM)==0):
            self.summary_dropped_NUM_ = self.summary_dropped_.iloc[0:0]
            self.summary_dropped_CAT_ = Summarize_Features(X, self.summary_dropped_, features_dropped_CAT, y)
        elif (len(self.features_dropped_)>0) & (len(features_dropped_CAT)==0):
            self.summary_dropped_CAT_ = self.summary_dropped_.iloc[0:0]
            self.summary_dropped_NUM_ = Summarize_Features(X, self.summary_dropped_, features_dropped_NUM, y)
        else:
            self.summary_dropped_NUM_ = Summarize_Features(X, self.summary_dropped_, features_dropped_NUM, y)
            self.summary_dropped_CAT_ = Summarize_Features(X, self.summary_dropped_, features_dropped_CAT, y)
        print(features_dropped_NUM, features_dropped_CAT)
        print('{} features with {} or fewer unique value(s)'.format(len(self.features_dropped_), self.unique_threshold))
        X = X[self.features_kept_]
        
        return self


    def transform(self, X, y=None):
        return X[self.features_kept_]

    

       

def Summarize_Features(df_X, df_Summary, ls_features, y=None):
    '''
    Purpose
    ---------------
    A custom utility function
    - (1) to append summary statistics and 
    - (2) to append churn rate if y is provided
    
    Parameters
    ---------------
    df_X: a pandas dataframe (df) with all features
    df_Summary: a df with simple summary statistics
    ls_features: a list of selected features
    y: a pandas series that represent a churn status
    - default: y=None

    Returns
    ---------------
    df_Summary_full: a df with full summary statistics
    '''

    df_desc         = df_X[ls_features].describe().T
    df_Summary_full = df_Summary. \
                      merge(df_desc, left_on='feature', right_index=True, how='inner').\
                      reset_index(drop=True)

    if not (y is None):    # When y is given
        temp_df    = pd.concat([y, df_X[ls_features]], axis=1)
        churn_rate = {}

        for fe in ls_features:
            y_mean = np.round(temp_df[(temp_df[fe].notnull())].iloc[:, 0].mean(), 4)
            churn_rate[fe] = y_mean

        churn_df   = pd.DataFrame.from_dict(churn_rate, orient='index')
        churn_df.columns = ['churn_rate']

        df_Summary_full = df_Summary. \
                          merge(churn_df, left_on='feature', right_index=True, how='inner').\
                          merge(df_desc, left_on='feature', right_index=True, how='inner').\
                          reset_index(drop=True)                

    return df_Summary_full


class RareCategoryEncoder(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer 
    - (1) to re-group rare categories into either 'rare_categories' or most common category
    - (2) to create more representative/manageable number of categories
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent a churn status
       - default: y=None
    category_min_pct: category Pct ~ [0, 1]
       - default: category_min_pct=0.01 after transformation
            - The % of any category be >= 1% of total sample.
            - Otherwise, this category is relabeled as either (1) 'rare_categories' or
              (2) the most common category after transformation.
    category_max_count: the number of categories/labels after transformation
       - default: category_count_threshold=20
            - The max number of categories after transformation <= 20
    encoding_method: how to re-classify rare categories
       - default: encoding_method=None --> assigned into 'rare_categories' category.
       - encoding_method='common' --> assigned into the most common category.
    
    Returns
    ---------------
    X_transformed: 
        - a transformed outcome in dataframe with proper column/index names
    category_mapper_: 
        - a data dictionary of mapping features to corresponding categories after encoding

    References
    ---------------        
    Feature Engine: 
        - https://pypi.org/project/feature-engine/
        - https://github.com/solegalli/feature_engine/blob/master/feature_engine/categorical_encoders.py
    '''

    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    

    def __init__(self,  features_CAT_ = None, category_min_pct=0.01, category_max_count=20, encoding_method=None, prefix=None, suffix=None):
        self.category_min_pct   = category_min_pct
        self.category_max_count = category_max_count
        self.encoding_method    = encoding_method
        self.prefix             = prefix
        self.suffix             = suffix
        self.features_CAT_      = features_CAT_
        # self.features_relevant  = features_relevant
        


    def fit(self, X, y=None):
        print('*'*50 + '\nPre-Processing: RareCategoryEncoder\n' + '*'*50 )
        # 0. Select Relevant Features for Rare Category Transformation
        # self.features_relevant.sort()
        # Note: 'category' is not intentionally included in dtypes.
        if self.features_CAT_ is None:
            self.features_CAT_            = X.select_dtypes(include=[object]).columns.tolist()

        # 1. Create Catetory Mapping Dictionary
        # - Total numer of categories = self.category_max_count
        # - Min pct of selected categories >= self.category_min_pct
        category_mapper_         = {}

        for fe in self.features_CAT_:
            X[fe] = X[fe].fillna('None')
            mapping              =  X[fe].value_counts(normalize=True).iloc[:self.category_max_count]
            category_mapper_[fe] = mapping[mapping >= self.category_min_pct].index

        # self.features_CAT_       = features_CAT_
        self.category_mapper_    = category_mapper_ 

        return self


    def transform(self, X, y=None):
        # encoding_method: 
        #   - default: 'rare_categories' for rare categories
        #   - if method = 'common', then the most common category for rare categories
        tmp_df             = X.copy()

        for fe in self.features_CAT_:
            if self.encoding_method is None:
                tmp_df[fe] =  np.where(tmp_df[fe].isin(self.category_mapper_[fe]), tmp_df[fe], 'rare_categories')
            elif self.encoding_method == 'common':
                tmp_df[fe] =  np.where(tmp_df[fe].isin(self.category_mapper_[fe]), \
                                       tmp_df[fe], self.category_mapper_[fe][0])

        # Note: A custom utility function, 'get_feature_name' is used.                                       
        tmp_df.columns     = get_feature_name(tmp_df, y=None, prefix=self.prefix, suffix=self.suffix)
        X_transformed      = tmp_df

        return X_transformed       

def get_feature_name(df_X, y=None, prefix=None, suffix=None):
    '''
    Purpose
    ---------------
    A custom utility function to provide appropriate feature names to transformed data
    
    Parameters
    ---------------
    df_X: a pandas dataframe (df)
    y: a pandas series that represent a churn status
        - default: y=None
    prefix: a string that is appended as prefix to the feature names of df_X
        - default: prefix=None
    suffix: a string that is appended as suffix to the feature names of df_X
        - default: suffix=None

    Returns
    ---------------
    feature_Name: a list of transformed feature names
    '''

    feature_Name     = df_X.columns.tolist()
    
    if (prefix is not None) & (suffix is not None):
        feature_Name = [prefix + '_' + fe + '_' + suffix for fe in df_X.columns]
    elif (prefix is not None):   
        feature_Name = [prefix + '_' + fe for fe in df_X.columns]
    elif (suffix is not None):   
        feature_Name = [fe + '_' + suffix for fe in df_X.columns]

    return feature_Name



class Num_Col_Covar(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer 
    - (1) to find the pairewise correlation amount all Num cols
    - (2) Then filter the pair that has high corr. coeff > 0.8
    - (3) Then drop one feature that has less lift 
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent dependent column
       - default: y=None
    c
    
    Returns
    ---------------
    X_transformed: 
        - a transformed outcome in dataframe with proper column/index names
    
    R
    '''

    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    

    def __init__(self, X, y, outcome_field = None ):
        
        self.X   = X
        self.y = y
        self.outcome_field = outcome_field
        
        
    def fit(self, X, y=None):
        print('*'*50 + '\nFeature Engineering: Num_Col_Covar\n' + '*'*50 )
        
        return self
        
        
    def transform(self, X, y=None):
        df = pd.concat([self.X, self.y], axis = 1)
        num_col = []
        cat_col = []
        for col in df.columns:
            if df[col].dtypes != 'object' and df[col].dtypes != 'string':
                num_col.append(col)
            else:
                cat_col.append(col)
        df[num_col] = df[num_col].fillna(0)
        corr_mat = round(df[num_col].corr(),1)
        corr_mat_stacked = corr_mat.stack().reset_index()
        corr_mat_stacked.columns = ['feature 1','feature 2','Correlation Coef']
        corr_mat_stacked = corr_mat_stacked[corr_mat_stacked['feature 1'] != corr_mat_stacked['feature 2']]
        corr_mat_stacked = corr_mat_stacked[corr_mat_stacked['Correlation Coef'] >= 0.8]


        #Splitting the churners and non churners to compute the averale lift for Numerical variables
        if self.outcome_field is not None:
            df_yes = df[df[self.outcome_field] == 1]
            df_no = df[df[self.outcome_field] == 0]
        
        else:
            print("Please give the Output/Dependent column")

        output_proportion = pd.DataFrame(columns = ['numerical feature', 'lift'])
        count = 0

        for col in num_col:
            # print("col:", col)
            output_mean, output_std = df_yes[col].mean(), df_yes[col].std()#This could be changed to upgrade_mean and noupgrade_mean
            no_output_mean, no_output_std = df_no[col].mean(), df_no[col].std()
            #print(churners_mean, no_churners_mean)
            print(no_output_mean)
            if no_output_mean != 0:
                lift_ = (output_mean-no_output_mean)/no_output_mean
                agg = col.split("_")[0]
            else:
                no_output_mean = 0.00001
                lift_ = (output_mean-no_output_mean)/no_output_mean
                agg = col.split("_")[0]


            output_proportion.loc[count] = [col,  lift_]
            count += 1

        # Creating 'feature_deletion' list
        feature_deletion = []
        for index, row in corr_mat_stacked.iterrows():   
            #print(row)
            lift_df = pd.DataFrame()
            lift_df['numerical feature'] = [row['feature 1'], row['feature 2']]
            # Merge with Correlation Coef
            lift_df = lift_df.merge(output_proportion[['numerical feature','lift']], on = 'numerical feature')

            lift_df['lift abs'] = np.abs(lift_df['lift'])
            lift_df = lift_df.sort_values(by = 'lift abs', ascending = True)
            display(lift_df)
            row_delete = lift_df.iloc[0] # droping the column that has low lift
            feature_delete = row_delete['numerical feature']
            print(feature_delete)

            feature_deletion += [feature_delete]


        # removing duplicates
        feature_deletion_final = []
        for i in feature_deletion:
            if i not in feature_deletion_final:
                feature_deletion_final.append(i)
        print("Deleted Num Features", len(feature_deletion_final),feature_deletion_final)

        #Updating Numerical column 
        num_col_1 = []
        for i in num_col:
            if i not in  feature_deletion_final:
                num_col_1.append(i)
        return  df[num_col_1 + cat_col]


class WOE_Encoding(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer 
    - (1) WOE for categorical columns 
    - (2) Then all categorical columns are replaced by WOE to become numerical columns 
    
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent dependent column
       - default: y=None
    c
    
    Returns
    ---------------
    X_transformed: 
        - a transformed outcome in dataframe with proper column/index names
    
    R
    '''

    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    
    

    def __init__(self,   y ):
        # self.X = X
        self.y = y
        
        
    def fit(self, X, y):
        print('*'*50 + '\nFeature Engineering: WOE Encoding for categorical features\n' + '*'*50 )
        
        return self
        
        
    def transform(self, X, y= None):
        import category_encoders as ce
        self.y = self.y.fillna(0)
        df = X.copy()
        columns = [col for col in df.columns if df[col].dtypes == 'object' ]
        num_col  = [col for col in df.columns if df[col].dtypes != 'object']
        woe_encoder = ce.WOEEncoder(cols=columns)
        woe_encoded_train = woe_encoder.fit_transform(df[columns], self.y).add_suffix('_woe')
        df = df.join(woe_encoded_train)

        woe_encoded_cols = woe_encoded_train.columns
        final_df = pd.concat([df[num_col], woe_encoded_train], axis = 1)
        df = final_df
        return df



    

class Vif_Feature_Select(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer 
    - (1) Calculate VIF for all the features 
    - (2) Then filter out all the features that has high VIF with a given threshold
    
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent dependent column
       - default: y=None
    c
    
    Returns
    ---------------
    X_transformed: 
        - a transformed outcome in dataframe with proper column/index names
    
    R
    '''

    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    

    def __init__(self,  VIF_threshold):
        self.VIF_threshold = VIF_threshold
        
        
    
    def fit(self, X, y):
        print('*'*50 + '\nFeature Engineering: VIF Feature Selection\n' + '*'*50 )
        
        return self
        
        
    def transform(self, X, y=None):
        from statsmodels.stats.outliers_influence import variance_inflation_factor 
        df = X.copy()
        df = df.fillna(0)
        for col in df.columns:
            df[col] = df[col].astype('float')
        from statsmodels.stats.outliers_influence import variance_inflation_factor 
        # VIF dataframe 
        vif_data = pd.DataFrame() 
        vif_data["feature"] = df.columns 

        # calculating VIF for each feature 
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                                  for i in range(len(df.columns))] 

        selected_col = list(vif_data[vif_data['VIF'] <= self.VIF_threshold]['feature'])
        
        
        return df[selected_col]


class Down_Sampling(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer 
    - (1) Down Sampling  
   
    
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent dependent column
       - default: y=None
    c
    
    Returns
    ---------------
    X_transformed: 
        - a transformed outcome in dataframe with equal number of 1s and 0s
    
    
    '''

    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    

    def __init__(self,  target):
        self.target = target
        
        
    
    def fit(self, X, y = None):
        print('*'*50 + '\nDown Sampling' + '*'*50 )
        
        return self
        
        
    def transform(self, X, y = None):
        df_input = X.copy()
        df_postive = df_input[df_input[self.target]==1]
        df_negative=df_input[df_input[self.target]==0]
        df_negative_downsample = resample(df_negative, replace = False, n_samples = df_postive.shape[0], random_state = 483)
        train_down = pd.concat([df_postive,df_negative_downsample]).sample(frac = 1, random_state = 1)
        X_train_down = train_down.drop(columns =self.target)
        y_train_down = train_down[self.target]
        # return pd.concat([X_train_down,y_train_down], axis = 1)
        return X_train_down,y_train_down





# class XGBoost_Model_Training(BaseEstimator, TransformerMixin):
#     '''
#     Purpose
#     ---------------
#     A custom transformer 
#     - (1) To Train the Model my XGBoost with all hyperm -parameter tunning and optimization 

    

#     Parameters
#     ---------------
#     X: a pandas dataframe (df) that consists of all possible features
#     y: a pandas series that represent dependent column
#        - default: y=None
#     best_features - selected best features (By default all X variables in the dataset)
#     cv - hyper parameter for cross validation (default is 3)

#     Returns
#     ---------------
#     best_model: 
#         - Trained XGBoost model with all optimal hyperparameters 


#     '''

#     # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary


#     def __init__(self, target, cv = 3):
#         self.target = target
#         self.cv = cv

        

#     def fit(self, X, y):
#         print('*'*50 + '\nXGBoost Model Training' + '*'*50 )

#         return self


#     def transform(self, X, y = None):
#         ####################
#         # 3, grid_search - get best hyper parameters
#         #    **** Input: X_train_down[best features], y_train_down
#         #    **** Output: best hyper parameters
#         print(X.shape)
#         y = X[self.target]
#         X = X.drop(columns = self.target)
#         X_train_down = X.copy()
#         y_train_down = y.copy()
#         ## remove the special character in the columns
#         X_train_down.columns = X_train_down.columns.str.split('.').str[0]

#         columns = []
#         for col in X_train_down.columns:
#             col = col.replace('>', 'greater_than')
#             col = col.replace('<', 'lesser_than')
#             columns.append(col)
#         X_train_down.columns = columns

#         # print(X_train_down.shape)
#         y_train_down = y

#         print('*'*50 + "\nget best hyper parameters\n" )
#         clf = xgb.XGBClassifier(objective = 'binary:logistic', n_jobs = -1)
#         parameters = {
#              "eta"              : [0.05, 0.10, 0.15, 0.20] ,
#              "max_depth"        : [ 3,5,7,9],
#              "min_child_weight" : [ 1, 3, 5],
#              "gamma"            : [ 0.0, 0.1, 0.2, 0.3],
#              "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7]

#              }

#         grid = GridSearchCV(clf, parameters, scoring="roc_auc", cv = self.cv)
#         start = time.time()
#         grid.fit(X_train_down, y_train_down)
#         best_model = grid.best_estimator_

#         best_model.fit(X_train_best_features, y_train_down)

#         # Training down_sample Error
#         y_train_pred = best_model.predict(X_train_best_features)
#         y_train_proba = best_model.predict_proba(X_train_best_features)[:,1]
#         train_acc, train_recall, train_f1 = acc(y_train_down,y_train_pred), recall(y_train_down,y_train_pred), \
#                                                         f1(y_train_down,y_train_pred)
#         train_fpr, train_tpr, train_thresholds = roc_curve(y_train_down, y_train_proba, pos_label=1)

#         # Training all Error
#         y_train_pred_all = best_model.predict(X_train_all_best_features)
#         y_train_proba_all = best_model.predict_proba(X_train_all_best_features)[:,1]
#         train_acc_all, train_recall_all, train_f1_all = acc(y_train,y_train_pred_all), recall(y_train,y_train_pred_all), \
#                                                         f1(y_train,y_train_pred_all)
#         train_fpr_all, train_tpr_all, train_thresholds_all = roc_curve(y_train, y_train_proba_all, pos_label=1)


#         # Testing Error
#         y_test_pred = best_model.predict(X_test_best_features)
#         y_test_proba = best_model.predict_proba(X_test_best_features)[:,1]
#         test_acc, test_recall, test_f1 = acc(y_test,y_test_pred), recall(y_test,y_test_pred), \
#                                                     f1(y_test,y_test_pred)
#         # all data get the probability
#         y_pred = best_model.predict(X_best_features)
#         y_proba = best_model.predict_proba(X_best_features)[:,1]

#         ## Metrics Summary
#         metrics_summary = pd.DataFrame()
#         metrics_summary['Training down_sample'] = [train_acc,train_recall,train_f1]
#         metrics_summary['Training'] = [train_acc_all,train_recall_all,train_f1_all]
#         metrics_summary['Testing'] = [test_acc,test_recall,test_f1]
#         metrics_summary.index = ['Accuracy','Recall','F1-Score']

#         # Roc AUC
#         test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_proba, pos_label=1)
#         plt.figure(figsize = (15,10))
#         plt.plot(train_fpr, train_tpr, color = 'blue', label = 'Train')
#         plt.plot(test_fpr, test_tpr, color = 'red', label = 'Test')
#         plt.plot([0,1], [0,1], color = 'black')
#         plt.xlabel('False Positive Rate', fontsize = 'x-large')
#         plt.ylabel('True Positive Rate', fontsize = 'x-large')
#         plt.title(f'ROC Curve for:', fontsize = 'x-large')
#         plt.legend(fontsize = 'large')
#         plt.show()

#         return best_model, metrics_summary, y_test_proba, y_train_proba_all,y_proba,y_pred 

    

def grid_search (X_train_down, y_train_down, ratio , cv = 3):
        ####################
        # 3, grid_search - get best hyper parameters
        #    **** Input: X_train_down[best features], y_train_down
        #    **** Output: best hyper parameters
        
        
        X_train_down.columns = X_train_down.columns.str.split('.').str[0]
        
        columns = []
        for col in X_train_down.columns:
            col = col.replace('>', 'greater_than')
            col = col.replace('<', 'lesser_than')
            columns.append(col)
        X_train_down.columns = columns
        
        
        print('*'*50 + "\nget best hyper parameters\n")
        import xgboost as xgb
        clf = xgb.XGBClassifier(objective = 'binary:logistic', n_jobs = -1,   scale_pos_weight=ratio, random_state  = 500)
        parameters = {
             # "eta"              : [0.05, 0.10, 0.15, 0.20] ,
             "eta"              : [0.05,  0.20] ,
             # "max_depth"        : [ 3,5,7,9],
             "max_depth"        : [ 3,9],
             # "min_child_weight" : [ 1, 3, 5, 10],
             "min_child_weight" : [1],
             # "gamma"            : [ 0.3, 0.5, 1, 1.5, 2, 5],
             "gamma"            : [ 0.0, 0.3],
             # "colsample_bytree" : [ 0.3, 0.5 , 0.7, 1.0],
            # "subsample": [0.6, 0.8, 1.0],
             "colsample_bytree" : [ 0.3, 0.7]
    
             }

        grid = GridSearchCV(clf, parameters, scoring="roc_auc", cv = cv)
        start = time.time()
        grid.fit(X_train_down, y_train_down)
        best_model = grid.best_estimator_
        
        return (best_model)


def model_evaluation (best_model,X_train_best_features, y_train_down, X_train_all_best_features, y_train, X_test_best_features, y_test, X_best_features, y):
        ####################
        # 4, model evaluation - evaluate model results
        # **** Input: best model,X_train_best_features, y_train_down, X_test_best_features, y_test
        # **** Output:  metrics_summary - accuracy, recall and f1
  
        best_model.fit(X_train_best_features, y_train_down)

        # Training down_sample Error
        y_train_pred = best_model.predict(X_train_best_features)
        y_train_proba = best_model.predict_proba(X_train_best_features)[:,1]
        train_acc, train_recall, train_precision, train_f1 = acc(y_train_down,y_train_pred), recall(y_train_down,y_train_pred), precision(y_train_down,y_train_pred),f1(y_train_down,y_train_pred)
        train_fpr, train_tpr, train_thresholds = roc_curve(y_train_down, y_train_proba, pos_label=1)
        
        # Training all Error
        y_train_pred_all = best_model.predict(X_train_all_best_features)
        y_train_proba_all = best_model.predict_proba(X_train_all_best_features)[:,1]
        train_acc_all, train_recall_all, train_precision_all, train_f1_all = acc(y_train,y_train_pred_all), recall(y_train,y_train_pred_all), precision(y_train,y_train_pred_all),\
                                                        f1(y_train,y_train_pred_all)
        train_fpr_all, train_tpr_all, train_thresholds_all = roc_curve(y_train, y_train_proba_all, pos_label=1)
        
        
        # Testing Error
        y_test_pred = best_model.predict(X_test_best_features)
        y_test_proba = best_model.predict_proba(X_test_best_features)[:,1]
        test_acc, test_recall, test_precision, test_f1 = acc(y_test,y_test_pred), recall(y_test,y_test_pred), precision(y_test,y_test_pred), \
                                                    f1(y_test,y_test_pred)
        # all data get the probability
        y_pred = best_model.predict(X_best_features)
        y_proba = best_model.predict_proba(X_best_features)[:,1]
        
        ## Metrics Summary
        metrics_summary = pd.DataFrame()
        metrics_summary['Training down_sample'] = [train_acc,train_recall,train_precision,train_f1]
        metrics_summary['Training'] = [train_acc_all,train_recall_all,train_precision_all,train_f1_all]
        metrics_summary['Testing'] = [test_acc,test_recall,test_precision,test_f1]
        metrics_summary.index = ['Accuracy','Recall','Precision','F1-Score']

        # Roc AUC
        test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_proba, pos_label=1)
        plt.figure(figsize = (15,10))
        plt.plot(train_fpr, train_tpr, color = 'blue', label = 'Train')
        plt.plot(test_fpr, test_tpr, color = 'red', label = 'Test')
        plt.plot([0,1], [0,1], color = 'black')
        plt.xlabel('False Positive Rate', fontsize = 'x-large')
        plt.ylabel('True Positive Rate', fontsize = 'x-large')
        plt.title(f'ROC Curve for:', fontsize = 'x-large')
        plt.legend(fontsize = 'large')
        plt.show()
        # Get the current figure
        # fig = plt.show()

        # Clear the current figure to prevent displaying it immediately
        plt.clf()

        
        return (metrics_summary, y_test_pred,  y_test_proba, y_train_proba_all,y_proba,y_pred)


def profiling_train(df_profile_train_df, target, shap_cat_col_train):
#     top_shap_features_train = list(shap_feature_importance_df_trainig['Fetaures'].head(ton_num).values)
#     shap_num_col_train = []
#     shap_cat_col_train = []
#     for col in top_shap_features_train:
#         if 'woe' in col:
#             temp_col = col.replace('_woe', '')
#             shap_cat_col_train.append(temp_col)
#         else:
#             shap_num_col_train.append(col)
        
    
    # This is to get the actual proportion
    normalized_actual_df_train = pd.DataFrame(columns = ['features', 'levels',  'actual_prop'])
    for col in shap_cat_col_train:
        # print(col)
        # print(pd.DataFrame(df_profile_train_df[[col, target]].value_counts(normalize = True)).reset_index())
        normalized_actual_df_train_temp = pd.DataFrame(df_profile_train_df[[col, target]].value_counts(normalize = True)).reset_index().rename(columns = { 0: 'actual_prop'}).sort_values(col)
        normalized_actual_df_train_temp['features'] = col
        normalized_actual_df_train_temp['levels'] = normalized_actual_df_train_temp[col]
        normalized_actual_df_train_temp = normalized_actual_df_train_temp[normalized_actual_df_train_temp['incomplete_flag'] == 1]
        normalized_actual_df_train_temp = normalized_actual_df_train_temp[['features', 'levels',  'actual_prop']]
        normalized_actual_df_train = pd.concat([normalized_actual_df_train, normalized_actual_df_train_temp], axis = 0)   
        
        
    # This is to get the observed proportion in Rank_1_5
    actual_df_train_Rank_1_5 = df_profile_train_df[df_profile_train_df['Rank'] < 6]
    normalized_actual_df_train_Rank_1_5 = pd.DataFrame(columns = ['features', 'levels',  'Observed_Rank_1_5_prop'])
    # normalized_actual_df_train = pd.DataFrame(columns = ['features', 'levels',  'Observed_Rank_1_5_prop'])
    for col in shap_cat_col_train:
        # print(col)
        # print(pd.DataFrame(actual_df_train_Rank_1_5[[col, target]].value_counts(normalize = True)).reset_index())
        normalized_actual_df_train_temp_Rank_1_5 = pd.DataFrame(actual_df_train_Rank_1_5[[col, target]].value_counts(normalize = True)).reset_index().rename(columns = { 0: 'Observed_Rank_1_5_prop'}).sort_values(col)
        normalized_actual_df_train_temp_Rank_1_5['features'] = col
        normalized_actual_df_train_temp_Rank_1_5['levels'] = normalized_actual_df_train_temp_Rank_1_5[col]
        normalized_actual_df_train_temp_Rank_1_5 = normalized_actual_df_train_temp_Rank_1_5[normalized_actual_df_train_temp_Rank_1_5['incomplete_flag'] == 1]
        normalized_actual_df_train_temp_Rank_1_5 = normalized_actual_df_train_temp_Rank_1_5[['features', 'levels',  'Observed_Rank_1_5_prop']]
        normalized_actual_df_train_Rank_1_5 = pd.concat([normalized_actual_df_train_Rank_1_5, normalized_actual_df_train_temp_Rank_1_5], axis = 0)
        
        
    # This is to get the observed proportion in Rank_6_10
    actual_df_train_Rank_6_10 = df_profile_train_df[(df_profile_train_df['Rank'] < 11) & (df_profile_train_df['Rank'] > 5)]
    normalized_actual_df_train_Rank_6_10 = pd.DataFrame(columns = ['features', 'levels',  'Observed_Rank_6_10_prop'])
    # normalized_actual_df_train = pd.DataFrame(columns = ['features', 'levels',  'Observed_Rank_6_10_prop'])
    for col in shap_cat_col_train:
        # print(col)
        # print(pd.DataFrame(actual_df_train_Rank_6_10[[col, target]].value_counts(normalize = True)).reset_index())
        normalized_actual_df_train_temp_Rank_6_10 = pd.DataFrame(actual_df_train_Rank_6_10[[col, target]].value_counts(normalize = True)).reset_index().rename(columns = { 0: 'Observed_Rank_6_10_prop'}).sort_values(col)
        normalized_actual_df_train_temp_Rank_6_10['features'] = col
        normalized_actual_df_train_temp_Rank_6_10['levels'] = normalized_actual_df_train_temp_Rank_6_10[col]
        normalized_actual_df_train_temp_Rank_6_10 = normalized_actual_df_train_temp_Rank_6_10[normalized_actual_df_train_temp_Rank_6_10['incomplete_flag'] == 1]
        normalized_actual_df_train_temp_Rank_6_10 = normalized_actual_df_train_temp_Rank_6_10[['features', 'levels',  'Observed_Rank_6_10_prop']]
        normalized_actual_df_train_Rank_6_10 = pd.concat([normalized_actual_df_train_Rank_6_10, normalized_actual_df_train_temp_Rank_6_10], axis = 0)
        
    # Merging actual and observed proportion into one dataframe
    normalized_actual_df_train = normalized_actual_df_train.merge(normalized_actual_df_train_Rank_1_5, on = ['features', 'levels'], how = 'inner')
    normalized_actual_df_train = normalized_actual_df_train.merge(normalized_actual_df_train_Rank_6_10, on = ['features', 'levels'], how = 'inner')
        
    # Find the lift feature vise
    normalized_actual_df_train['riskiest 1-5% vs. total (Index)'] = normalized_actual_df_train['Observed_Rank_1_5_prop'] / normalized_actual_df_train['actual_prop']
    normalized_actual_df_train['riskiest 6-10% vs. total (Index)'] = normalized_actual_df_train['Observed_Rank_6_10_prop'] / normalized_actual_df_train['actual_prop']
        
        
    # This is to get the color for the output based on the risk profile
    def highlight_rows(row): 
        #value = row.loc['churners vs. total (Index)']
        value1 = row.loc['riskiest 1-5% vs. total (Index)']
        value2 = row.loc['riskiest 6-10% vs. total (Index)']

        if  value1 >= 2.4 or value2 >= 2.4:
            color = '#FF0000' # Bold Red
        elif  (value1 >= 1.2 and value1 < 2.4) or (value2 >= 1.2 and value2 < 2.4):
            color = '#FFB3BA' # Red
        elif  value1 <= 0.4 or value2 <= 0.4:
            color = '#00CD00' # Bold Green        
        elif  (value1 <= 0.8 and value1 > 0.4) or (value2 <= 0.8 and value2 > 0.4):
            color = '#BAFFC9' # Green
        else:
            color = '#FFFFFF' # White
        return ['background-color: {}'.format(color) for r in row]
    normalized_actual_df_train['colour'] = 'colour'
        
        
    # This is to assign colour for each row based on the risk profile
    for i in range(0, normalized_actual_df_train.shape[0]):
        value1 = float(normalized_actual_df_train.iloc[i:i+1,5])
        value2 = float(normalized_actual_df_train.iloc[i:i+1,6])
        if(value1 >= 2.4 or value2 >= 2.4):
            normalized_actual_df_train['colour'].iloc[i] = 'Bold Red - Extreme Risk' #FF0000' # Bold Red
        elif  (value1 >= 1.2 and value1 < 2.4) or (value2 >= 1.2 and value2 < 2.4):
            normalized_actual_df_train['colour'].iloc[i] = 'Red - High Risk' #FFB3BA' # Red
        elif value1 <= 0.4 or value2 <= 0.4:
            normalized_actual_df_train['colour'].iloc[i] = 'Bold Green - Extreme Low Risk' #00CD00' # Bold Green        
        elif  (value1 <= 0.8 and value1 > 0.4) or (value2 <= 0.8 and value2 > 0.4):
            normalized_actual_df_train['colour'].iloc[i] = 'Green - Low Risk' #BAFFC9' # Green
        else:
            normalized_actual_df_train['colour'].iloc[i] = 'White - Neutral Risk' #FFFFFF' # White

        
    # The final colour coded train dataframe
    color_train_df = normalized_actual_df_train.style.apply(highlight_rows, axis=1)
        
    return color_train_df


def ks(data=None,target=None, prob=None):
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], 10)
    grouped = data.groupby('bucket', as_index = False)
    # print(grouped.min())
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events']   = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = kstable['cum_eventrate']-kstable['cum_noneventrate'] 

    #Formating
    kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    pd.set_option('display.max_columns', 9)
    # print(kstable)
    
    #Display KS
    # from colorama import Fore
    # print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return kstable


class OHE_Encoding(BaseEstimator, TransformerMixin):
    '''
    Purpose
    ---------------
    A custom transformer 
    - (1) OHE for categorical columns 
    - (2) Then all categorical columns are replaced by WOE to become numerical columns 
    
    
    Parameters
    ---------------
    X: a pandas dataframe (df) that consists of all possible features
    y: a pandas series that represent dependent column
       - default: y=None
    c
    
    Returns
    ---------------
    X_transformed: 
        - a transformed outcome in dataframe with proper column/index names
    
    R
    '''

    # Features that are irrelevant to analysis: TYPE in ['DNU', 'TARGET'] from Data Dictionary
    
    

    def __init__(self,   X, model_type = "Train" ):
        self.X = X
        # self.cat_cols = cat_cols
        self.model_type = model_type
        
        
    def fit(self, X, y = None):
        print('*'*50 + '\nFeature Engineering: OHE Encoding for categorical features\n' + '*'*50 )
        
        return self
        
        
    def transform(self, X, y= None):
        cat_cols = []
        import category_encoders as ce
        from sklearn.preprocessing import OneHotEncoder
        
        
        
        cat_cols = []
        num_cols = []
        for col in X.columns:
            if X[col].dtypes == 'object' or X[col].dtypes == 'string[python]':
                cat_cols.append(col)
            else:
                num_cols.append(col)
        def custom_combiner(feature, category):
            return str(feature) + "_" + type(category).__name__ + "_" + str(category)
        print('Before OHE encoding:', X.shape)
        X[cat_cols] = X[cat_cols].astype('str')
        if self.model_type == 'Train':
            ohe = OneHotEncoder(
                            drop = 'first',
                            max_categories = 10, feature_name_combiner='concat')
        else:
            ohe = OneHotEncoder(
                            # drop = 'first',
                            max_categories = 10, feature_name_combiner='concat')
        temp_enc_df = ohe.fit_transform(X[cat_cols]).toarray()
        out_df = X[num_cols].copy()
        out_df[list(ohe.get_feature_names_out())] = temp_enc_df
        # temp_enc_df = pd.get_dummies(self.X[cat_cols], drop_first = True)
        # print('After OHE encoding:', temp_enc_df.shape)

#         self.y = self.y.fillna(0)
#         df = X.copy()
#         columns = [col for col in df.columns if df[col].dtypes == 'object']
#         num_col  = [col for col in df.columns if df[col].dtypes != 'object']
#         woe_encoder = ce.WOEEncoder(cols=columns)
#         woe_encoded_train = woe_encoder.fit_transform(df[columns], self.y).add_suffix('_woe')
#         df = df.join(woe_encoded_train)

#         woe_encoded_cols = woe_encoded_train.columns
#         final_df = pd.concat([df[num_col], woe_encoded_train], axis = 1)
#         df = final_df
        return out_df

def lift_analysis(y_test, y_test_proba, y_test_pred):
    ####################
    # 5, lift analysis - get lift curve for a df
    #    **** Input: y_test, y_test_proba
    #    **** Output: df with lift and also share of churn

    y_test_arr = np.array(y_test).reshape(-1,1)
    y_test_proba_arr = y_test_proba.reshape(-1,1)
    y_test_pred_arr = y_test_pred.reshape(-1,1)
    y_test_concat = np.concatenate((y_test_arr, y_test_proba_arr, y_test_pred_arr), axis = 1)
    y_test_concat = pd.DataFrame(y_test_concat, columns = ['True Label','Predicted Score',  'Predicted label'])
    y_test_concat = y_test_concat.sort_values(by = 'Predicted Score', ascending = False)
    nb_obs = y_test_concat.shape[0] + 1
    intervals = np.linspace(0,nb_obs,11).astype('int')
    x_axis = [f'{k*10}% - {(k+1)*10}%' for k in range(10)]

    churn_rate_overall = y_test.mean()
    churn_ratio = []
    churners=[]
    non_churners = []
    total = []
    #decile = []
    true_positive_list = []
    true_negative_list = []
    false_positive_list = []
    false_negative_list = []
    precision_list = []
    recall_list = []
    predicted_positive_list = []

    for ind in range(10):
        y_sub = y_test_concat.iloc[intervals[ind]:intervals[ind+1]]
        prec = precision(y_sub['True Label'], y_sub['Predicted label'])
        reca = recall(y_sub['True Label'], y_sub['Predicted label'])
        precision_list.append(prec)
        recall_list.append(reca)
        # print(y_sub)
        ## churn Rate
        churn = y_sub['True Label'].sum()/y_sub.shape[0]
        ratio = churn/churn_rate_overall
        churn_ratio += [ratio]
        total += [y_sub['True Label'].shape[0]]
        churners+=[y_sub['True Label'].sum()]
        
        # print(y_sub['True Label'].shape[0])
        # print(y_sub['True Label'].sum())
        non_churners += [y_sub['True Label'].shape[0] - y_sub['True Label'].sum()]
        
        
        
        
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        predicted_positive = 0
        
        for i in range(0, y_sub.shape[0]):
            predicted_positive += y_sub.iloc[i,2].sum()
            # print(y_sub.iloc[i,0], y_sub.iloc[i,2])
            if (y_sub.iloc[i,0] == 1) & (y_sub.iloc[i,2] == 1):
                # print('hi')
                true_positive += 1
                # print(true_positive)
            elif (y_sub.iloc[i,0] == 0) & (y_sub.iloc[i,2] == 0):
                true_negative += 1
            elif (y_sub.iloc[i,0] == 1) & (y_sub.iloc[i,2] == 0):
                false_negative += 1
            elif (y_sub.iloc[i,0] == 0) & (y_sub.iloc[i,2] == 1):
                false_positive += 1

        true_positive_list.append(true_positive)
        true_negative_list.append(true_negative)
        false_negative_list.append(false_negative)
        false_positive_list.append(false_positive)
        predicted_positive_list.append(predicted_positive)

        
        
        
        
        
    lift_df = pd.DataFrame()
    lift_df['Decile'] = x_axis
    lift_df['Lift'] = churn_ratio
    lift_df['total'] = total
    lift_df['share of events']=churners
    lift_df['share of non events']=non_churners
    lift_df['captured event rate'] = churners/lift_df['share of events'].sum()
   
    
    print(churners)
    print(total)
    print(non_churners)
    # print(lift_df['share of Incomplete Flag'].sum())
    # print(lift_df)
    
    lift_df['captured non event rate'] = non_churners/lift_df['share of non events'].sum()
    lift_df['cumsum event rate'] = lift_df['captured event rate'].cumsum()
    lift_df['cumsum non event rate'] = lift_df['captured non event rate'].cumsum()
    lift_df['KS'] = lift_df['cumsum event rate'] - lift_df['cumsum non event rate']


#     lift_df['predicted_positive'] = predicted_positive_list
#     lift_df['true_positive'] = true_positive_list
#     lift_df['true_negative'] = true_negative_list
#     lift_df['false_negative'] = false_negative_list
#     lift_df['false_positive'] = false_positive_list

#     lift_df['precision'] = precision_list
#     lift_df['recall'] = recall_list


    return (lift_df)


def lift_analysis(y_test, y_test_proba):
        ####################
        # 5, lift analysis - get lift curve for a df
        #    **** Input: y_test, y_test_proba
        #    **** Output: df with lift and also share of churn

        y_test_arr = np.array(y_test).reshape(-1,1)
        y_test_proba_arr = y_test_proba.reshape(-1,1)
        y_test_concat = np.concatenate((y_test_arr, y_test_proba_arr), axis = 1)
        y_test_concat = pd.DataFrame(y_test_concat, columns = ['True Label','Predicted Score'])
        y_test_concat = y_test_concat.sort_values(by = 'Predicted Score', ascending = False)
        nb_obs = y_test_concat.shape[0] + 1
        intervals = np.linspace(0,nb_obs,11).astype('int')
        x_axis = [f'{k*10}% - {(k+1)*10}%' for k in range(10)]

        churn_rate_overall = y_test.mean()
        churn_ratio = []
        churners=[]
        #decile = []
        for ind in range(10):
            y_sub = y_test_concat.iloc[intervals[ind]:intervals[ind+1]]
            ## churn Rate
            churn = y_sub['True Label'].sum()/y_sub.shape[0]
            ratio = churn/churn_rate_overall
            churn_ratio += [ratio]
            churners+=[y_sub['True Label'].sum()]
        lift_df = pd.DataFrame()
        lift_df['Decile'] = x_axis
        lift_df['Lift'] = churn_ratio
        lift_df['share of churn']=churners
        return (lift_df)
def lift_analysis_5(y_test, y_test_proba):
    
    
            ####################
            # 5, lift analysis - get lift curve for a df
            #    **** Input: y_test, y_test_proba
            #    **** Output: df with lift and also share of churn

    y_test_arr = np.array(y_test).reshape(-1,1)
    y_test_proba_arr = y_test_proba.reshape(-1,1)
    y_test_concat = np.concatenate((y_test_arr, y_test_proba_arr), axis = 1)
    y_test_concat = pd.DataFrame(y_test_concat, columns = ['True Label','Predicted Score'])
    y_test_concat = y_test_concat.sort_values(by = 'Predicted Score', ascending = False)
    nb_obs = y_test_concat.shape[0] + 1
    intervals = np.linspace(0,nb_obs,21).astype('int')
    x_axis = [f'{k*5}% - {(k+1)*5}%' for k in range(20)]

    churn_rate_overall = y_test.mean()
    churn_ratio = []
    churners=[]
    #decile = []
    for ind in range(20):
        y_sub = y_test_concat.iloc[intervals[ind]:intervals[ind+1]]
        ## churn Rate
        churn = y_sub['True Label'].sum()/y_sub.shape[0]
        ratio = churn/churn_rate_overall
        churn_ratio += [ratio]
        churners+=[y_sub['True Label'].sum()]
    lift_df = pd.DataFrame()
    lift_df['Decile'] = x_axis
    lift_df['Lift'] = churn_ratio
    lift_df['share of churn']=churners
    return (lift_df)

def lift_analysis_1(y_test, y_test_proba):
            ####################
            # 1, lift analysis - get lift curve for a df
            #    **** Input: y_test, y_test_proba
            #    **** Output: df with lift and also share of churn

    y_test_arr = np.array(y_test).reshape(-1,1)
    y_test_proba_arr = y_test_proba.reshape(-1,1)
    y_test_concat = np.concatenate((y_test_arr, y_test_proba_arr), axis = 1)
    y_test_concat = pd.DataFrame(y_test_concat, columns = ['True Label','Predicted Score'])
    y_test_concat = y_test_concat.sort_values(by = 'Predicted Score', ascending = False)
    nb_obs = y_test_concat.shape[0] + 1
    intervals = np.linspace(0,nb_obs,101).astype('int')
    x_axis = [f'{k*1}% - {(k+1)*1}%' for k in range(100)]

    churn_rate_overall = y_test.mean()
    churn_ratio = []
    churners=[]
    #decile = []
    for ind in range(100):
        y_sub = y_test_concat.iloc[intervals[ind]:intervals[ind+1]]
        ## churn Rate
        churn = y_sub['True Label'].sum()/y_sub.shape[0]
        ratio = churn/churn_rate_overall
        churn_ratio += [ratio]
        churners+=[y_sub['True Label'].sum()]
    lift_df = pd.DataFrame()
    lift_df['Decile'] = x_axis
    lift_df['Lift'] = churn_ratio
    lift_df['share of churn']=churners
    return (lift_df)

def lift_analysis_final(y_train, y_train_proba, y_test, y_test_proba, y, y_proba):
    lift_train = lift_analysis(y_train, y_train_proba)
    lift_test  = lift_analysis(y_test, y_test_proba)
    lift_all   = lift_analysis(y, y_proba)


    #### break down lift curve with interval as 5 in the top 20 decile
    lift_train_ = lift_analysis_5(y_train, y_train_proba)
    lift_test_  = lift_analysis_5(y_test, y_test_proba)
    lift_all_   = lift_analysis_5(y, y_proba)


    #### break down lift curve with interval as 1 in the top 20 decile
    lift_train_1 = lift_analysis_1(y_train, y_train_proba)
    lift_test_1  = lift_analysis_1(y_test, y_test_proba)
    lift_all_1   = lift_analysis_1(y, y_proba)


    lift_train_final = pd.concat([lift_train_.head(4),lift_train.tail(8)],axis=0).rename(columns={'Lift':'train_lift','share of churn':'train_share of churn'})
    lift_test_final  = pd.concat([lift_test_.head(4),lift_test.tail(8)],axis=0).rename(columns={'Lift':'test_lift','share of churn':'test_share of churn'})
    lift_all_final   = pd.concat([lift_all_.head(4),lift_all.tail(8)],axis=0).rename(columns={'Lift':'overall_lift','share of churn':'all_share of churn'})

    lift_train_final['train_perc_actual_churn'] = lift_train_final['train_share of churn'] / lift_train_final['train_share of churn'].sum()
    lift_test_final['test_perc_actual_churn'] = lift_test_final['test_share of churn'] / lift_test_final['test_share of churn'].sum()
    lift_all_final['all_perc_actual_churn'] = lift_all_final['all_share of churn'] / lift_all_final['all_share of churn'].sum()


    lift_train_final_detailed = pd.concat([lift_train_1[0:5], lift_train_[1:4],lift_train.tail(8)],axis=0).rename(columns={'Lift':'train_lift','share of churn':'train_share of churn'})
    lift_test_final_detailed  = pd.concat([lift_test_1[0:5], lift_test_[1:4],lift_test.tail(8)],axis=0).rename(columns={'Lift':'test_lift','share of churn':'test_share of churn'})
    lift_all_final_detailed   = pd.concat([lift_all_1[0:5], lift_all_[1:4],lift_all.tail(8)],axis=0).rename(columns={'Lift':'overall_lift','share of churn':'all_share of churn'})

    lift_train_final_detailed['train_perc_actual_churn'] = lift_train_final_detailed['train_share of churn'] / lift_train_final_detailed['train_share of churn'].sum()
    lift_test_final_detailed['test_perc_actual_churn'] = lift_test_final_detailed['test_share of churn'] / lift_test_final_detailed['test_share of churn'].sum()
    lift_all_final_detailed['all_perc_actual_churn'] = lift_all_final_detailed['all_share of churn'] / lift_all_final_detailed['all_share of churn'].sum()


    lift_actual = lift_train_final.merge(lift_test_final, on='Decile',how='inner')\
                    .merge(lift_all_final,on='Decile',how='inner')

    lift_per_decile = lift_train_final_detailed.merge(lift_test_final_detailed, on='Decile',how='inner')\
                    .merge(lift_all_final_detailed,on='Decile',how='inner')
    
    return lift_actual, lift_per_decile

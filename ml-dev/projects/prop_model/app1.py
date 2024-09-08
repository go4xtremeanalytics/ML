import streamlit as st
import pandas as pd


import streamlit as st
from streamlit_shap import st_shap
import shap

# Your SHAP code here to generate shap_values

# Display the plot
st_shap(shap.plots.waterfall(shap_values[0]))
st.write("""
# My first app
Churn Prediction Model helps to identify the high risk customers who will be likely to churn in the coming days. This app is scheduled to refresh every day with the new updated list of active customers and their predicted probability results for churning
""")
rank_df = pd.read_csv("rank_df.csv")

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
            """)
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model



st.title('CHURN PREDICTION')
st.write("""
Created by Ridhwan Muttaqien - HCK06


         
Use the sidebar to input customer data.
""")
@st.cache_data
def fetch_data():
    df = pd.read_csv('churn.csv')
    return df

df = fetch_data()

transaction = st.number_input('avg_transaction_value', 0.0)
points = st.number_input('points_in_wallet', 0.0)
membership = st.selectbox('membership_category', df['membership_category'].unique())
feedback = st.selectbox('feedback', df['feedback'].unique())


data = {
    'avg_transaction_value': transaction,
    'points_in_wallet': points,
    'membership_category': membership,
    'feedback': feedback,
    }
input = pd.DataFrame(data, index=[0])

st.subheader('Customer Data')
st.write(input)

transform = joblib.load("prepro.pkl")
load_model = load_model("ann_churn.h5")

if st.button('Predict'):
    ubah = transform.transform(input)
    prediction = load_model.predict(ubah)
    res_pred = np.where(prediction >= 0.5, 1, 0)


    if res_pred == 1:
        res_pred = 'Churned'
    else:
        res_pred = 'Stayed'

    st.write('Based on Customer Data, the customer predicted: ')
    st.write(res_pred)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import model_train

st.title('HR Practice Web')
st.subheader('HR')

model, features, X, Y = model_train()

st.sidebar.header(
    'Input'
)

annual_income = st.sidebar.slider(
    'Annual Income',
    max_value = 1000000,
    min_value = 1000,
    value = 100,
    step=1
)

dti = st.sidebar.slider(
    'DTI',
    max_value = 40.5,
    min_value = 1.0,
    value = 0.25,
    step=0.1
)

int_rate = st.sidebar.slider(
    'Interest Rate',
    max_value = 50.0,
    min_value = 1.0,
    value = 3.0,
    step=0.1
)

total_acc = st.sidebar.slider(
    'Total Account',
    max_value = 20,
    min_value = 1,
    value = 3,
    step = 1
)

if st.sidebar.button('Predict Loan Amount'):
    input_features = pd.DataFrame(
        [[annual_income, dti, int_rate, total_acc]],columns = features
    )

    prediction = model.predict(input_features)[0]

    st.success('Prediction Successfull.')
    st.metric(
        label = "Predicted Loan Amount",
        value = f"Rs. {prediction:,.2f}"
    )
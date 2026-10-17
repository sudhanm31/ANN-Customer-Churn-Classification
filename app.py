import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf

model = tf.keras.models.load_model('ann_model.h5')

#load the encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as f:
    encoder_geo = pickle.load(f)

with open('le_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", encoder_geo.categories_[0])
gender = st.selectbox("Gender", le_gender.classes_)
age = st.slider("Age", min_value=18, max_value=90)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", min_value=0, max_value=10)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],      
        })

geo_encoded = encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
prediction_probability = prediction[0][0]

if prediction_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")

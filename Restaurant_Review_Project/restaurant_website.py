import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(layout="wide")

scaler = joblib.load(open('Restaurant_Review_Project/scaler.pkl','rb'))

st.title("Restaurant Rating Prediction App")

st.divider()

averagecsot = st.number_input("Please enter the estimated average cost for two", min_value=50, max_value=999999, value=1000, step=200)

tablebooking = st.selectbox("Restaurant has table booking?", ["Yes", "No"])

onlinedelivery = st.selectbox("Restaurant has online delivery?", ["Yes", "No"])

pricerange = st.selectbox("What is the price range? (1 Cheapest, 4 Most expensive)", [1,2,3,4])

predictbutton = st.button("Predict the review")

st.divider()

model = joblib.load(open('Restaurant_Review_Project/mlmodel.pkl', 'rb'))

bookingstatus = 1 if tablebooking == "Yes" else 0

deliverystatus = 1 if onlinedelivery == "Yes" else 0

values = [[averagecsot, bookingstatus, deliverystatus, pricerange]]
my_x_values = np.array(values)

X = scaler.transform(my_x_values)

if predictbutton:
    
    prediction = model.predict(X)

    st.write(prediction)

    if prediction < 2.5:
        st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.0:
        st.write("Good")
    elif prediction < 4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")


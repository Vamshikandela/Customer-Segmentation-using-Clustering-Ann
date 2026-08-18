from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load preprocess + model
with open("preprocess.pkl", 'rb') as f:
    preprocess = pickle.load(f)

ann_model = load_model('ann_model.keras')

st.title("Customer Engagement Segmentation Using Clustering & ANN")
st.image("customer_engaged.png")
# Load dataset (for dropdown values)
df = pd.read_csv("customers.csv")

# Inputs
unit_price = st.number_input('Enter unit_price', min_value=1.0)

quantity = st.number_input("Select quantity",min_value=1.0)

discount_percent = st.slider(
    "Select discount_percent",
    min_value=1,max_value=70,
)

page_views = st.number_input("Enter pages viewed", min_value=1)

time_on_site_sec = st.number_input("Enter time on site", min_value=1)
session_duration_bucket = st.selectbox(
    "Session Duration Bucket",
    ['Very Short', 'Short', 'Long', 'Very Long']
)

# Label mapping (VERY IMPORTANT)
label_map = {
    0: "High_Engaged_User",
    1: "Low_Engaged_User",
    2: "Occasional_Engaged_User"
}

def model():
    # Create proper input row
    input_data = pd.DataFrame([{
        "unit_price": float(unit_price),
        "quantity": quantity,
        "discount_percent":float(discount_percent),
        "pages_viewed": page_views,
        "time_on_site_sec":float(time_on_site_sec),
        "session_duration_bucket":session_duration_bucket

    }])

    # Preprocess
    data = preprocess.transform(input_data)

    # Predict (softmax output)
    pred = ann_model.predict(data)

    # Convert probabilities  (class index)
    pred_class = np.argmax(pred, axis=1)[0]

    return pred_class,pred


if st.button('Predict'):
    pred_class, pred_probs = model()

    label = label_map[pred_class]

    #st.success(f"Predicted Segment: {label}")
    if label=="High_Engaged_User":
        st.write("Congrats! You Have Won 1000 worth Vochers Kindly Redeem It")
        st.image("high_engaged.png")
    elif label =="Low_Engaged_User":
        st.write("Congrats! You Got  Flat 10% Discount on Next Order")
        st.image("low_engaged.png")
    elif label=="Occasional_Engaged_User":
        st.write("Congrats! You Have Won 500 worth Vochers Kindly Redeem It")
        st.image("occasion_engaged.png")

    #st.write("Prediction Probabilities:", pred_probs)
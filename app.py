import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('crop_yield_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load original data for label encoders
try:
    data = pd.read_csv('/content/yield_df.csv')  # Modify path if needed
except FileNotFoundError:
    st.error("Error: 'yield_df.csv' not found. Please make sure it's uploaded to Colab.")
    st.stop()  # Stop app execution if file not found

# Create and fit label encoders
le_country = LabelEncoder()
le_item = LabelEncoder()
le_country.fit(data['Area'])
le_item.fit(data['Item'])

def predict_yield(country, item, pesticides, avg_temp, rainfall):
    """Predicts crop yield based on input features."""
    input_data = pd.DataFrame({
        'Country_Encoded': [le_country.transform([country])[0]],
        'Item_Encoded': [le_item.transform([item])[0]],
        'Pesticides': [pesticides],
        'Avg_Temp': [avg_temp],
        'Rainfall': [rainfall]
    })
    prediction = model.predict(input_data)[0]
    return prediction

# Streamlit app
st.title("Crop Yield Prediction")

country = st.selectbox("Country", data['Area'].unique())
item = st.selectbox("Crop Item", data['Item'].unique())
pesticides = st.number_input("Pesticides (tonnes)", value=100.0)
avg_temp = st.number_input("Average Temperature (°C)", value=25.0)
rainfall = st.number_input("Rainfall (mm/year)", value=1000.0)

if st.button("Predict"):
    predicted_yield = predict_yield(country, item, pesticides, avg_temp, rainfall)
    st.success(f"Predicted Yield for {item} in {country}: {predicted_yield:.2f} hg/ha")

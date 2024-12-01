# import streamlit as st
# import pandas as pd
import numpy as np
# import joblib  # For loading the trained model

# # Load pre-trained model
# model = joblib.load('LinearRegressionModel.pkl')  # Replace with your model file


# brand_dict = {'Nissan': 0, 'alfa-romero': 1, 'audi': 2, 'bmw': 3, 'buick': 4, 'chevrolet': 5, 'dodge': 6, 'honda': 7, 'isuzu': 8, 'jaguar': 9, 'maxda': 10, 'mazda': 11, 'mercury': 12, 'mitsubishi': 13, 'nissan': 14, 'peugeot': 15, 'plymouth': 16, 'porcshce': 17, 'porsche': 18, 'renault': 19, 'saab': 20, 'subaru': 21, 'toyota': 22, 'toyouta': 23, 'vokswagen': 24, 'volkswagen': 25, 'volvo': 26, 'vw': 27}
# fueltype_dict = {'diesel': 0, 'gas': 1}

# # App title
# st.title("Car Price Prediction")

# # Sidebar for input features
# st.sidebar.header("Car Details")

# # Input fields
# brand = st.sidebar.selectbox("Make", ["Toyota", "Honda", "BMW", "Audi", "Ford"])
# if brand in brand_dict:
#     brand_encoded = brand_dict[brand]
#     print(f"Encoded value for '{brand}': {brand_encoded}")
# else:
#     print(f"Category '{brand}' not found in mapping.")
# # car_model = st.sidebar.text_input("Model", "Corolla")
# # car_year = st.sidebar.slider("Year", 2000, 2023, 2015)
# citympg= st.sidebar.number_input("Mileage (in km)", value=50000, step=1000)
# highwaympg = st.sidebar.number_input("hMileage (in km)", value=50000, step=1000)

# fueltype = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
# if fueltype in fueltype_dict:
#     fueltype_encoded = fueltype_dict[fueltype]
#     print(f"Encoded value for '{fueltype}': {fueltype_encoded}")
# else:
#     print(f"Category '{fueltype}' not found in mapping.")

# # transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

# # Process user input into model input
# input_data = pd.DataFrame({
#     'brand': [brand_encoded],
#     'fueltype': [fueltype_encoded],
#     'citympg': [citympg],
#     'highwaympg': [highwaympg],
#     # 'model': [car_model],
#     # 'year': [car_year],
#     # 'transmission': [transmission]
# })

# # Dummy encoding for categorical variables (if necessary)
# # Adjust this part based on your model's preprocessing
# # input_data = pd.get_dummies(input_data, columns=['make', 'fuel_type', 'transmission'], drop_first=True)

# # Predict
# if st.sidebar.button("Predict Price"):
#     prediction = model.predict(input_data)[0]  # Get the prediction
#     st.write(f"### Predicted Price: ₹{prediction:,.2f}")

# # Footer
# st.write("This tool is a demonstration for educational purposes.")




import streamlit as st
import pandas as pd
import joblib  # For loading the trained model

# Load pre-trained model`
model = joblib.load('cloud.pkl')

# Dictionaries for encoding
brand_dict = {'nissan': 0, 'alfa-romero': 1, 'audi': 2, 'bmw': 3, 'buick': 4, 'chevrolet': 5, 'dodge': 6, 'honda': 7, 'isuzu': 8, 'jaguar': 9, 'mazda': 10, 'mercury': 12, 'mitsubishi': 13, 'peugeot': 15, 'plymouth': 16, 'porsche': 18, 'renault': 19, 'saab': 20, 'subaru': 21, 'toyota': 22, 'volkswagen': 25, 'volvo': 26}
fueltype_dict = {'diesel': 0, 'gas': 1}

# App title
st.title("Car Price Prediction")

# Sidebar for input features
st.sidebar.header("Car Details")

# Inputs
brand = st.sidebar.selectbox("Make", list(brand_dict.keys()))
brand_encoded = brand_dict.get(brand.lower(), -1)
if brand_encoded == -1:
    st.error(f"Brand '{brand}' is not recognized.")

citympg = st.sidebar.number_input("City MPG", value=25, step=1)
highwaympg = st.sidebar.number_input("Highway MPG", value=30, step=1)

fueltype = st.sidebar.selectbox("Fuel Type", ["Diesel", "Gas"])
fueltype_encoded = fueltype_dict.get(fueltype.lower(), -1)
if fueltype_encoded == -1:
    st.error(f"Fuel type '{fueltype}' is not recognized.")

# Prepare input data
input_data = pd.DataFrame({
    'brand': [brand_encoded],
    'fueltype': [fueltype_encoded],
    'citympg': [citympg],
    'highwaympg': [highwaympg]
})

# Predict
try:
    if st.sidebar.button("Predict Price"):
        prediction = np.abs(model.predict(input_data)[0])
        st.write(f"### Predicted Price: ₹{prediction:,.2f}")
except Exception as e:
    st.error(f"Error in prediction: {e}")

# Footer
st.write("This tool is a demonstration for educational purposes.")

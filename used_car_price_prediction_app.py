import streamlit as st
import pandas as pd
import joblib

# --- Load the model and encoders ---
loaded_model = joblib.load('rf_trainedmodel.pkl')
loaded_encoders = joblib.load('label_encoders.pkl')
loaded_scaler = joblib.load('min.pkl')

categorical_columns = [
    "Body_type",
    "Transmission",
    "Fuel_type",
    "Location",
]

# --- Prediction Function ---
@st.cache_resource
def predict_price(
    mileage,
    engine_displacement,
    year_of_manufacture,
    transmission,
    fuel_type,
    owner_no,
    model_year,
    location,
    kilometer_driven,
    body_type,
):

    # Create a DataFrame from the input values
    input_data = pd.DataFrame(
        {
            "Mileage": [mileage],
            "Engine_displacement": [engine_displacement],
            "Year_of_manufacture": [year_of_manufacture],
            "Transmission": [transmission],
            "Fuel_type": [fuel_type],
            "Owner_No.": [owner_no],
            "Model_year": [model_year],
            "Location": [location],
            "Kilometer_Driven": [kilometer_driven],
            "Body_type": [body_type],
        }
    )

    # Encode categorical features using the loaded LabelEncoders
    for col in categorical_columns:
        if col in input_data.columns:
            le = loaded_encoders[col]
            input_data[col] = le.transform(input_data[col].astype(str))

    # Make the prediction
    predicted_price = loaded_model.predict(input_data)
    predicted_price_norm = loaded_scaler.inverse_transform([[predicted_price[0]]])[0][0]
    return predicted_price_norm


# --- Streamlit UI ---
st.set_page_config(page_title="Used Car Price Prediction", page_icon=":red_car:", layout="centered")
st.title(":red_car: **Used Car Price Prediction**")
st.write("Enter the details of the used car to estimate its resale price.")

# Input fields in a form layout
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        mileage = st.number_input("Mileage (kmpl)", min_value=0.0, format="%.1f")
        engine_displacement = st.number_input("Engine Displacement (cc)", min_value=0)
        year_of_manufacture = st.number_input(
            "Year of Manufacture", min_value=1900, max_value=2024
        )
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    with col2:
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
        owner_no = st.number_input("Number of Previous Owners", min_value=0)
        model_year = st.number_input("Model Year", min_value=1900, max_value=2024)
        kilometer_driven = st.number_input("Kilometers Driven", min_value=0)

    location = st.selectbox(
        "Location",
        ["Chennai", "Bangalore", "Delhi", "Kolkata", "Jaipur", "Hyderabad"],
    )

    body_type = st.selectbox(
        "Body Type",
        [
            "Hatchback",
            "SUV",
            "Sedan",
            "MUV",
            "Minivans",
            "Coupe",
            "Pickup Trucks",
            "Convertibles",
            "Hybrids",
            "Wagon",
        ],
    )

    # Submit button
    submitted = st.form_submit_button("Estimate Price")

# Display the prediction result
if submitted:
    try:
        predicted_price = predict_price(
            mileage,
            engine_displacement,
            year_of_manufacture,
            transmission,
            fuel_type,
            owner_no,
            model_year,
            location,
            kilometer_driven,
            body_type,
        )
        st.success(f"Estimated Price: ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

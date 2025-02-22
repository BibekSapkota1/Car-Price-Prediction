import joblib
import numpy as np
import pandas as pd

# Load the trained model
# model = joblib.load("lgb_model.pkl")  # Light Gradient-Boosting Model
# model = joblib.load("ridge_model.pkl")  # Ridge Regression Model
model = joblib.load("xgboost_model.pkl")  # Ridge Regression Model

# Load encoders
brand_encoder = joblib.load("brand_encoder.pkl")
print(type(brand_encoder))
fuel_type_encoder = joblib.load("fuel_type_encoder.pkl")
transmission_encoder = joblib.load("transmission_encoder.pkl")
ext_col_encoder = joblib.load("ext_col_encoder.pkl")
int_col_encoder = joblib.load("int_col_encoder.pkl")
accident_encoder = joblib.load("accident_encoder.pkl")
clean_title_encoder = joblib.load("clean_title_encoder.pkl")

# Define input values (same as your Streamlit app)
input_data = {
    "brand": "Mercedes-Benz",
    "mileage": 7388,
    "fuel_type": "Gasoline",
    "transmission": "Automatic",
    "ext_col": "Black",
    "int_col": "Beige",
    "accident": "None reported",
    "clean_title": "Yes",
    "horsepower": 208,
    "displacement": 2,
    "cylinder_count": 4,
    "model_age": 4
}

# Function to safely transform categorical values
def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return -1  # Assign a default value for unseen labels

# Encode categorical features
encoded_features = [
    safe_transform(brand_encoder, input_data["brand"]),
    input_data["mileage"],
    safe_transform(fuel_type_encoder, input_data["fuel_type"]),
    safe_transform(transmission_encoder, input_data["transmission"]),
    safe_transform(ext_col_encoder, input_data["ext_col"]),
    safe_transform(int_col_encoder, input_data["int_col"]),
    safe_transform(accident_encoder, input_data["accident"]),
    safe_transform(clean_title_encoder, input_data["clean_title"]),
    input_data["horsepower"],
    input_data["displacement"],
    input_data["cylinder_count"],
    input_data["model_age"]
]

# Column names (updated to match the exact ones in your model's training data)
feature_columns = [
    'brand', 'milage', 'fuel_type', 'transmission', 'ext_col', 'int_col', 
    'accident', 'clean_title', 'Horsepower', 'Displacement', 'Cylinder Count', 'model_age'
]

# Convert the features_array to a pandas DataFrame
features_df = pd.DataFrame([encoded_features], columns=feature_columns)

# Check for missing columns (based on the model's expected input)
expected_columns = feature_columns  # Use the column names you used during training
missing_columns = set(expected_columns) - set(features_df.columns)

# If any columns are missing, add them with default values (e.g., 0 or -1)
for col in missing_columns:
    features_df[col] = 0  # You can use a default value, such as 0, for numerical columns

# Make prediction
predicted_price = model.predict(features_df)

# Print results
print("\n🔍 Final Feature Array for Prediction:")
print(features_df)
print(f"\n💰 Predicted Car Price: ${predicted_price[0]:,.2f}")





#   ----------------------------- do not modify this -----------------------------
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime

# # Load models
# models = {
#     "XGBoost": joblib.load("xgboost_model.pkl"),
#     "CatBoost": joblib.load("catboost_model.pkl"),
#     "LG Boost": joblib.load("lgb_model.pkl"),
#     "Aada Boost": joblib.load("adaboost_model.pkl"),
#     "Ridge Regression": joblib.load("ridge_model.pkl"),
# }

# # Load encoders
# brand_encoder = joblib.load("brand_encoder.pkl")
# fuel_type_encoder = joblib.load("fuel_type_encoder.pkl")
# transmission_encoder = joblib.load("transmission_encoder.pkl")
# ext_col_encoder = joblib.load("ext_col_encoder.pkl")
# int_col_encoder = joblib.load("int_col_encoder.pkl")
# accident_encoder = joblib.load("accident_encoder.pkl")
# clean_title_encoder = joblib.load("clean_title_encoder.pkl")

# # Monitoring CSV file
# LOG_FILE = "model_performance.csv"

# # Function to log model performance
# def log_performance(model_name, predicted_price, actual_price=None):
#     """Logs predictions to a CSV file with optional actual price."""
#     error = abs(predicted_price - actual_price) if actual_price else None
#     log_data = pd.DataFrame({
#         "timestamp": [datetime.now()],
#         "model": [model_name],
#         "predicted_price": [predicted_price],
#         "actual_price": [actual_price],
#         "error": [error]
#     })
    
#     # Append to CSV (create file if it doesn't exist)
#     log_data.to_csv(LOG_FILE, mode="a", header=not pd.io.common.file_exists(LOG_FILE), index=False)

# # Function to update actual prices in CSV
# def update_actual_price(index, actual_price):
#     """Updates the actual price in the CSV and recalculates error."""
#     if pd.io.common.file_exists(LOG_FILE):
#         df = pd.read_csv(LOG_FILE)
#         if 0 <= index < len(df):
#             df.at[index, "actual_price"] = actual_price
#             df.at[index, "error"] = abs(df.at[index, "predicted_price"] - actual_price)
#             df.to_csv(LOG_FILE, index=False)
#             st.success(f"✅ Actual price updated for entry {index}!")

# # Streamlit UI
# st.set_page_config(layout="wide")
# st.title("🚗 Car Price Prediction & Monitoring")

# # Sidebar for model selection and monitoring
# st.sidebar.header("Options")
# model_choice = st.sidebar.radio("Choose a model:", ["Main Model (Average)"] + list(models.keys()))
# show_monitoring = st.sidebar.checkbox("📊 Show Model Performance")

# # Input fields
# st.subheader("🔢 Enter Car Details")
# col1, col2 = st.columns(2)

# with col1:
#     brand = st.selectbox("Brand", brand_encoder.classes_)
#     mileage = st.number_input("Total Km Travelled (in km)", min_value=0)
#     fuel_type = st.selectbox("Fuel Type", fuel_type_encoder.classes_)
#     transmission = st.selectbox("Transmission", transmission_encoder.classes_)
#     ext_col = st.selectbox("Exterior Color", ext_col_encoder.classes_)
#     int_col = st.selectbox("Interior Color", int_col_encoder.classes_)

# with col2:
#     accident = st.selectbox("Accident History", accident_encoder.classes_)
#     clean_title = st.selectbox("Clean Title", clean_title_encoder.classes_)
#     horsepower = st.number_input("Horsepower (HP)", min_value=0)
#     displacement = st.number_input("Displacement (in L)", min_value=0.0)
#     cylinder_count = st.number_input("Cylinder Count", min_value=1)
#     model_age = st.number_input("Model Age (in years)", min_value=0)

# # Predict button
# if st.button("🔍 Predict Price"):
#     # Encode categorical features
#     encoded_features = [
#         brand_encoder.transform([brand])[0], mileage,
#         fuel_type_encoder.transform([fuel_type])[0],
#         transmission_encoder.transform([transmission])[0],
#         ext_col_encoder.transform([ext_col])[0],
#         int_col_encoder.transform([int_col])[0],
#         accident_encoder.transform([accident])[0],
#         clean_title_encoder.transform([clean_title])[0],
#         horsepower, displacement, cylinder_count, model_age
#     ]
    
#     # Convert features into a DataFrame
#     feature_columns = ['brand', 'milage', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title', 'Horsepower', 'Displacement', 'Cylinder Count', 'model_age']
#     features_df = pd.DataFrame([encoded_features], columns=feature_columns)
    
#     # Make prediction
#     if model_choice == "Main Model (Average)":
#         predictions = [model.predict(features_df)[0] for model in models.values()]
#         predicted_price = np.mean(predictions)
#     else:
#         predicted_price = models[model_choice].predict(features_df)[0]
    
#     # Display result
#     st.success(f"💰 Predicted Car Price: ${predicted_price:,.2f}")
    
#     # Log the prediction
#     log_performance(model_choice, predicted_price)

# # Monitoring Section
# if show_monitoring:
#     st.subheader("📊 Model Performance Monitoring")

#     if pd.io.common.file_exists(LOG_FILE):
#         df = pd.read_csv(LOG_FILE)

#         # Show raw data
#         st.subheader("🔍 Logged Predictions")
#         st.dataframe(df.tail(10))

#         # Allow user to enter actual price
#         st.subheader("📝 Enter Actual Price for a Prediction")
#         index_to_update = st.number_input("Entry Index to Update", min_value=0, max_value=len(df)-1, step=1)
#         actual_price_input = st.number_input("Actual Car Price ($)", min_value=0.0)

#         if st.button("✅ Update Actual Price"):
#             update_actual_price(index_to_update, actual_price_input)

#         # Plot error over time (if actual values exist)
#         if "error" in df.columns and df["error"].notna().any():
#             st.subheader("📈 Model Error Over Time")
#             st.line_chart(df.set_index("timestamp")["error"])
#     else:
#         st.warning("No logged predictions yet.")


"""
SuperKart Sales Forecasting - Streamlit Frontend
Calls the Flask backend API (hosted on a separate Hugging Face Space) to get predictions.
"""

import streamlit as st
import requests
import os

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="centered")

# Base URL of the Flask backend
BACKEND_URL = os.environ.get("BACKEND_URL") or "http://backend:7860/predict"

# Set the title of the Streamlit app
st.title("🛒 SuperKart Sales Forecast")
st.write("Predict the expected sales revenue for a product at a given store outlet.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Product details")
        product_weight = st.number_input("Product Weight", min_value=0.0, max_value=30.0, value=12.5, step=0.1)
        product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_area = st.number_input("Product Allocated Area (ratio)", min_value=0.0, max_value=0.3, value=0.05, step=0.001, format="%.3f")
        product_type = st.selectbox("Product Type", [
            "Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy", "Household",
            "Baking Goods", "Canned", "Health and Hygiene", "Meat", "Soft Drinks",
            "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood"
        ])
        product_mrp = st.number_input("Product MRP", min_value=0.0, max_value=300.0, value=150.0, step=1.0)
        product_category = st.selectbox("Product Category (from Product Id prefix)", ["Food", "Non-Consumable", "Drinks"])

    with col2:
        st.subheader("Store details")
        store_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2025, value=2005, step=1)
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])

    submitted = st.form_submit_button("Predict Sales")

if submitted:
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar,
        "Product_Allocated_Area": product_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Establishment_Year": store_year,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_city_type,
        "Store_Type": store_type,
        "Product_Id_Category": product_category,
    }
    try:
        with st.spinner("Getting forecast..."):
            resp = requests.post(BACKEND_URL, json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            st.success(f"### Predicted Sales Revenue: ₹{result['predicted_sales']:,.2f}")
        else:
            st.error(f"API error: {resp.text}")
    except Exception as e:
        st.error(f"Could not reach backend API: {e}")

st.caption("Model: Random Forest Regressor trained on SuperKart historical sales data.")

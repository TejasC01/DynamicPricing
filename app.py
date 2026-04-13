import streamlit as st
import pandas as pd
import numpy as np
from src.predict import predict_demand

st.set_page_config(page_title="Smart Demand Predictor", layout="wide")

# ------------------------------
# TITLE + EXPLANATION
# ------------------------------
st.title("📦 Smart Demand Predictor")

st.markdown("""
This tool helps you estimate how many products you will sell.

### How it works:
- Lower price → Higher demand  
- Near expiry → Faster sales  
- Weekend → Higher demand  

👉 Use this to decide the right price and avoid unsold stock.
""")

# ------------------------------
# INPUT SECTION
# ------------------------------
st.subheader("Enter Product Details")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Selling Price (₹)", min_value=1, max_value=1000, value=100)
    stock = st.number_input("Units in Stock", min_value=1, max_value=500, value=100)

with col2:
    expiry = st.number_input("Days before Expiry", min_value=0, max_value=30, value=5)

    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4,
        "Saturday": 5, "Sunday": 6
    }

    day_text = st.selectbox("Day of Week", list(day_map.keys()))
    day = day_map[day_text]

# ------------------------------
# PREDICTION BUTTON
# ------------------------------
if st.button("Predict Demand"):

    demand = predict_demand(price, stock, expiry, day)
    remaining = stock - demand

    st.subheader("📊 Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Expected Sales", f"{demand} units")

    with col2:
        st.metric("Unsold Stock", f"{max(0, remaining)} units")

    # ------------------------------
    # INSIGHTS (ACTIONABLE)
    # ------------------------------
    st.subheader("💡 Insights")

    if demand < stock * 0.5:
        st.warning("Demand is low → You may be pricing too high.")
    elif demand > stock * 0.9:
        st.success("Demand is very high → You could increase price slightly.")
    else:
        st.info("Demand is balanced.")

    # ------------------------------
    # GRAPH (FAST + MEANINGFUL)
    # ------------------------------
    st.subheader("📈 Price vs Demand")

    st.caption("See how changing price affects expected sales")

    price_range = np.linspace(max(10, price - 50), price + 50, 20)

    demand_values = [
        predict_demand(p, stock, expiry, day)
        for p in price_range
    ]

    graph_df = pd.DataFrame({
        "Price": price_range,
        "Expected Sales": demand_values
    })

    st.line_chart(graph_df.set_index("Price"))
import streamlit as st
import time
import pandas as pd

# Fix 9 & 2: Safeguard for missing artifacts and error handling
def load_backend_resources():
    try:
        # Mocking the loading of model/scaler
        model_exists = True # Change to False to test the error UI
        if not model_exists:
            raise FileNotFoundError("Model or Scaler artifacts not found.")
        return True
    except Exception as e:
        st.error(f"⚠️ Backend Error: {e}")
        return False

# Fix 10, 5, & 8: Data validation, batch capability, and interpretation
def get_prediction(price, stock, expiry, day):
    # Fix 3: Loading/Processing feedback
    with st.spinner('Predicting demand...'):
        time.sleep(1) # Simulate backend processing time
        
        # Mocking backend call
        # Fix 4: UI-level checks for unrealistic values
        raw_prediction = 45.6789 
        
        # Fix 8: Interpretation layer (Rounding and formatting)
        final_prediction = round(raw_prediction, 2)
        
        # Fix 4: Check if prediction > stock
        if final_prediction > stock:
            st.warning(f"Note: Predicted demand ({final_prediction}) exceeds current stock ({stock}).")
            return stock
        
        return final_prediction

# --- UI START ---
st.title("📈 Dynamic Pricing: Demand Prediction")

# Initialize Session State (Fix 6: State control)
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

resources_loaded = load_backend_resources()

if resources_loaded:
    # Sidebar Fix 10: Specific inputs and validation
    st.sidebar.header("Input Features")
    price = st.sidebar.slider("Product Price", 10.0, 500.0, 50.0)
    stock = st.sidebar.number_input("Current Stock Level", 0, 1000, 100)
    expiry = st.sidebar.slider("Days to Expiry", 0, 30, 7)
    
    # Fix 10: Enforcing specific day format for backend expectations
    day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day = st.sidebar.selectbox("Day of the Week", day_options)

    # Fix 7: User action separation (Separate buttons)
    col_btn1, col_btn2 = st.sidebar.columns(2)
    
    with col_btn1:
        if st.button("Predict Demand"):
            # Fix 6: Prediction only runs on button click
            st.session_state.prediction = get_prediction(price, stock, expiry, day)

    with col_btn2:
        if st.button("Generate Graph"):
            # Fix 5: Loop-based calls for demand vs price graph
            st.info("Generating batch inference visualization...")
            prices = [price * i for i in [0.8, 0.9, 1.0, 1.1, 1.2]]
            demands = [get_prediction(p, stock, expiry, day) for p in prices]
            st.line_chart(pd.DataFrame({"Price": prices, "Demand": demands}).set_index("Price"))

    # Display Results
    st.subheader("Results")
    if st.session_state.prediction is not None:
        # Fix 8: Categorize result
        status = "High Demand" if st.session_state.prediction > 50 else "Stable Demand"
        st.metric(label="Predicted Units Sold", value=f"{st.session_state.prediction} units", delta=status)
    else:
        st.write("Enter features and click 'Predict Demand' to see results.")
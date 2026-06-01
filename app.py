import streamlit as st
import pandas as pd
import altair as alt
from src.inference import predict_demand_from_df
from src.data_generation import DEFAULT_SAVE_PATH, generate_data


DAY_LABELS = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv(DEFAULT_SAVE_PATH)
    except FileNotFoundError:
        df = generate_data(save=True)

    df = df.copy()
    df["day_name"] = df["day_of_week"].map(DAY_LABELS)
    return df

# Set page config
st.set_page_config(page_title="Dynamic Pricing Project", layout="wide")

# Title and Description [cite: 85, 134]
st.title("📈 Dynamic Pricing: Demand Prediction")
st.markdown("### Phase 1: Machine Learning Inference Dashboard")

# Sidebar for Inputs [cite: 179]
st.sidebar.header("Input Features")
price = st.sidebar.slider("Product Price", 10.0, 500.0, 50.0)
stock = st.sidebar.number_input("Current Stock Level", 0, 1000, 100)
expiry = st.sidebar.slider("Days to Expiry", 0, 30, 7)
day = st.sidebar.selectbox("Day of the Week", 
                           ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# Convert UI inputs into a production DataFrame matching training feature names
day_mapping = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}

production_input = pd.DataFrame([{
    "price": price,
    "stock_level": stock,
    "days_to_expiry": expiry,
    "day_of_week": day_mapping[day]
}])

# Main Display Area [cite: 180, 181]
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Results")
    prediction = predict_demand_from_df(production_input)
    st.metric(label="Predicted Units Sold", value=round(prediction, 2))

with col2:
    st.subheader("Dataset Summary")
    demand_data = load_dataset()
    st.metric(label="Generated Dataset Rows", value=len(demand_data))
    st.metric(label="Average Units Sold", value=round(demand_data["units_sold"].mean(), 2))

st.divider()
st.subheader("Demand Patterns")

demand_data = load_dataset()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    price_chart = (
        alt.Chart(demand_data)
        .mark_circle(size=55, opacity=0.65)
        .encode(
            x=alt.X("price:Q", title="Price"),
            y=alt.Y("units_sold:Q", title="Units Sold"),
            color=alt.Color("days_to_expiry:Q", title="Days to Expiry"),
            tooltip=[
                "product_id:N",
                alt.Tooltip("price:Q", format=".2f"),
                "stock_level:Q",
                "days_to_expiry:Q",
                "day_name:N",
                "units_sold:Q",
            ],
        )
        .properties(height=340, title="Price vs Demand")
    )
    st.altair_chart(price_chart, use_container_width=True)

with chart_col2:
    day_order = list(DAY_LABELS.values())
    weekday_data = (
        demand_data.groupby("day_name", as_index=False)["units_sold"]
        .mean()
        .rename(columns={"units_sold": "avg_units_sold"})
    )
    weekday_chart = (
        alt.Chart(weekday_data)
        .mark_bar()
        .encode(
            x=alt.X("day_name:N", sort=day_order, title="Day of Week"),
            y=alt.Y("avg_units_sold:Q", title="Average Units Sold"),
            tooltip=[
                "day_name:N",
                alt.Tooltip("avg_units_sold:Q", format=".2f"),
            ],
        )
        .properties(height=340, title="Average Demand by Day")
    )
    st.altair_chart(weekday_chart, use_container_width=True)

expiry_data = (
    demand_data.groupby("days_to_expiry", as_index=False)["units_sold"]
    .mean()
    .rename(columns={"units_sold": "avg_units_sold"})
)
expiry_chart = (
    alt.Chart(expiry_data)
    .mark_line(point=True)
    .encode(
        x=alt.X("days_to_expiry:O", title="Days to Expiry"),
        y=alt.Y("avg_units_sold:Q", title="Average Units Sold"),
        tooltip=[
            "days_to_expiry:O",
            alt.Tooltip("avg_units_sold:Q", format=".2f"),
        ],
    )
    .properties(height=320, title="Demand by Expiry Window")
)
st.altair_chart(expiry_chart, use_container_width=True)

st.success("UI Branch Initialized - Ready for Integration")

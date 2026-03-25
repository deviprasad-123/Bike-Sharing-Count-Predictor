import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Rental Predictor",
    page_icon="🚲",
    layout="centered"
)

# ── Load model bundle ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "bikemodel.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'bikemodel.pkl' not found. Please run the notebook first to generate it.")
        st.stop()
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle      = load_model()
model       = bundle["model"]
transformer = bundle["transformer"]
model_name  = bundle["model_name"]
feature_cols = bundle["feature_cols"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚲 Bike Rental Count Predictor")
st.markdown(f"Predict the **total number of bike rentals** for a given day using **{model_name}**.")
st.divider()

# ── Sidebar — model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.success(f"**Active Model:** {model_name}")
    st.markdown("**Features used:**")
    for col in feature_cols:
        st.markdown(f"- `{col}`")
    st.divider()
    st.caption("Train the model by running the fixed notebook and saving `bikemodel.pkl`.")

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📋 Enter Day Details")

col1, col2 = st.columns(2)

with col1:
    season = st.selectbox(
        "Season",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "🌱 Spring", 2: "☀️ Summer", 3: "🍂 Fall", 4: "❄️ Winter"}[x]
    )

    yr = st.selectbox(
        "Year",
        options=[0, 1],
        format_func=lambda x: "2011" if x == 0 else "2012"
    )

    holiday = st.selectbox(
        "Holiday",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    weekday = st.selectbox(
        "Day of Week",
        options=list(range(7)),
        format_func=lambda x: ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"][x]
    )

with col2:
    workingday = st.selectbox(
        "Working Day",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    weathersit = st.selectbox(
        "Weather Situation",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "☀️ Clear / Few clouds",
            2: "🌥️ Mist / Cloudy",
            3: "🌧️ Light Snow / Rain",
            4: "⛈️ Heavy Rain / Storm"
        }[x]
    )

    temp = st.slider("Temperature (Normalized)", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                     help="0 = coldest, 1 = hottest")

    atemp = st.slider("Feeling Temperature (Normalized)", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                      help="0 = coldest feel, 1 = hottest feel")

windspeed = st.slider("Wind Speed (Normalized)", min_value=0.0, max_value=1.0, value=0.2, step=0.01,
                      help="0 = calm, 1 = maximum recorded speed")

# ── Build input DataFrame matching training feature order ─────────────────────
input_dict = {
    "season":     season,
    "yr":         yr,
    "holiday":    holiday,
    "weekday":    weekday,
    "workingday": workingday,
    "weathersit": weathersit,
    "temp":       temp,
    "atemp":      atemp,
    "windspeed":  windspeed,
}

# Only keep / reorder columns that were used during training
input_df = pd.DataFrame([input_dict])[feature_cols]

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()

if st.button("🔮 Predict Bike Rentals", use_container_width=True, type="primary"):
    input_scaled = transformer.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    prediction   = max(0, round(prediction))   # rentals can't be negative

    st.success(f"### 🚲 Predicted Rentals: **{prediction:,}** bikes")

    # Context bar
    if prediction < 1500:
        level, color = "Low demand day", "🔵"
    elif prediction < 3500:
        level, color = "Moderate demand day", "🟡"
    elif prediction < 5500:
        level, color = "High demand day", "🟠"
    else:
        level, color = "Very high demand day", "🔴"

    st.info(f"{color} **{level}** — plan bike availability accordingly.")

    # Show the input summary
    with st.expander("📊 Input Summary"):
        summary = input_df.copy()
        summary.columns = feature_cols
        st.dataframe(summary, use_container_width=True)


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
os.chdir("C:/PROJECTS/HOUSE PRICE PREDICTION")


# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main {
    background-color: #f7f4ef;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}

.hero h1 {
    font-size: 2.6rem;
    margin: 0;
    letter-spacing: -1px;
    color: #f0e6d3;
}

.hero p {
    color: #a0aec0;
    margin-top: 0.5rem;
    font-size: 1rem;
    font-weight: 300;
}

.section-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #0f3460;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
}

.result-box {
    background: linear-gradient(135deg, #0f3460, #1a1a2e);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    color: white;
}

.result-box .label {
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #a0aec0;
    margin-bottom: 0.4rem;
}

.result-box .price {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: #f0e6d3;
    font-weight: 700;
}

.result-box .range {
    font-size: 0.85rem;
    color: #718096;
    margin-top: 0.3rem;
}

.tag {
    display: inline-block;
    background: #e8f4f8;
    color: #0f3460;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px;
}

.stButton > button {
    background: linear-gradient(135deg, #0f3460, #1a1a2e) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: #4a5568 !important;
}

.footer {
    text-align: center;
    color: #a0aec0;
    font-size: 0.78rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e2d9ce;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load('Models/house_price_model.pkl')
    scaler   = joblib.load('Models/house_price_scaler.pkl')
    features = joblib.load('Models/house_price_features.pkl')
    return model, scaler, features

try:
    model, scaler, features = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏠 House Price Predictor</h1>
    <p>Fill in the details below to get an instant price estimate</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model files. Make sure the `Models/` folder exists.\n\n`{load_error}`")
    st.stop()


# ─── Input Form ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Property Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=500, max_value=20000,
                            value=3000, step=100)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
    stories = st.selectbox("Stories", [1, 2, 3, 4], index=1)

with col2:
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
    parking = st.selectbox("Parking Spots", [0, 1, 2, 3], index=1)
    furnishing = st.selectbox("Furnishing Status",
                               ["Furnished", "Semi-Furnished", "Unfurnished"], index=0)

st.markdown('<div class="section-label">Amenities</div>', unsafe_allow_html=True)

col3, col4, col3b = st.columns(3)
with col3:
    airconditioning = st.toggle("Air Conditioning", value=True)
    mainroad = st.toggle("Main Road Access", value=True)
with col4:
    prefarea = st.toggle("Preferred Area", value=False)
    basement = st.toggle("Basement", value=False)
with col3b:
    guestroom = st.toggle("Guest Room", value=False)
    hotwaterheating = st.toggle("Hot Water Heating", value=False)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("Estimate Price →")


# ─── Prediction ─────────────────────────────────────────────────────────────
if predict_btn:
    # Build input dict
    input_data = {
        'area':                             area,
        'bedrooms':                         bedrooms,
        'bathrooms':                        bathrooms,
        'stories':                          stories,
        'mainroad':                         int(mainroad),
        'guestroom':                        int(guestroom),
        'basement':                         int(basement),
        'hotwaterheating':                  int(hotwaterheating),
        'airconditioning':                  int(airconditioning),
        'prefarea':                         int(prefarea),
        'parking':                          parking,
        'furnishingstatus_semi-furnished':  int(furnishing == "Semi-Furnished"),
        'furnishingstatus_unfurnished':     int(furnishing == "Unfurnished"),
    }

    input_df = pd.DataFrame([input_data])

    # Auto-fill any missing columns
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[features]

    # Scale & predict
    input_scaled    = scaler.transform(input_df)
    predicted_price = model.predict(input_scaled)[0]

    low  = predicted_price * 0.90
    high = predicted_price * 1.10

    st.markdown(f"""
    <div class="result-box">
        <div class="label">Estimated Price</div>
        <div class="price">₨ {predicted_price:,.0f}</div>
        <div class="range">Likely range: ₨ {low:,.0f} – ₨ {high:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    # Summary tags
    st.markdown("<br>", unsafe_allow_html=True)
    tags = [
        f"🏠 {area} sq ft",
        f"🛏 {bedrooms} bed",
        f"🚿 {bathrooms} bath",
        f"🏗 {stories} stor{'y' if stories==1 else 'ies'}",
        f"🚗 {parking} parking",
        f"🛋 {furnishing}",
    ]
    if airconditioning: tags.append("❄️ AC")
    if prefarea:        tags.append("⭐ Preferred Area")
    if basement:        tags.append("🏚 Basement")
    if guestroom:       tags.append("🛎 Guest Room")

    tags_html = "".join([f'<span class="tag">{t}</span>' for t in tags])
    st.markdown(f"<div style='text-align:center'>{tags_html}</div>", unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Powered by Gradient Boosting Regressor &nbsp;·&nbsp; For estimation purposes only
</div>
""", unsafe_allow_html=True)
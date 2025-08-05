import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data and train model
data = pd.read_csv("advertising.csv")
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
model = LinearRegression().fit(X, y)

# Page configuration
st.set_page_config(page_title="Sales Predictor", page_icon="📈", layout="centered")

# Custom title with subheading
st.markdown("<h1 style='text-align: center;'>📊 Advertising Sales Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Estimate your sales based on your marketing budget</p>", unsafe_allow_html=True)
st.markdown("---")

# Input Section
st.header("🧮 Enter Advertising Budget")
col1, col2, col3 = st.columns(3)

with col1:
    tv = st.number_input("💡 TV Budget ($)", min_value=0.0, value=100.0, step=10.0)

with col2:
    radio = st.number_input("📻 Radio Budget ($)", min_value=0.0, value=50.0, step=5.0)

with col3:
    newspaper = st.number_input("🗞️ Newspaper Budget ($)", min_value=0.0, value=30.0, step=5.0)

# Predict button
if st.button("🔮 Predict Sales"):
    prediction = model.predict([[tv, radio, newspaper]])
    st.success(f"📈 Estimated Sales: **{prediction[0]:.2f}** units")

# Footer
st.markdown("---")
st.caption("🚀 Built with Streamlit | Model: Linear Regression")

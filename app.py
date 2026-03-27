import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add src to path
sys.path.append('src')
from preprocessing import impute_missing, encode_quality, feature_engineering

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load train data to get means
train = pd.read_csv('data/train.csv')
train_p = feature_engineering(encode_quality(impute_missing(train)))
X_raw = train_p.drop(['Id', 'SalePrice'], axis=1)
X_raw_dummies = pd.get_dummies(X_raw, drop_first=True)
means = X_raw_dummies.mean()

st.title("🏠 House Price Prediction")

st.markdown("Entrez les caractéristiques de la maison pour prédire le prix.")

# User inputs
lot_area = st.number_input("Superficie du terrain (Lot Area)", value=10000, min_value=1000, max_value=100000)
year_built = st.number_input("Année de construction (Year Built)", value=2000, min_value=1800, max_value=2026)
overall_qual = st.slider("Qualité générale (Overall Quality)", 1, 10, 5)
gr_liv_area = st.number_input("Surface habitable au sol (Ground Living Area)", value=1500, min_value=500, max_value=5000)
full_bath = st.slider("Salles de bain complètes (Full Bathrooms)", 0, 5, 2)
garage_cars = st.slider("Places de garage (Garage Cars)", 0, 4, 2)

if st.button("Prédire le prix"):
    # Create input df
    input_df = pd.DataFrame([means], columns=X_raw_dummies.columns)
    
    # Set user values
    input_df['LotArea'] = lot_area
    input_df['YearBuilt'] = year_built
    input_df['OverallQual'] = overall_qual
    input_df['GrLivArea'] = gr_liv_area
    input_df['FullBath'] = full_bath
    input_df['GarageCars'] = garage_cars
    
    # Scale
    input_scaled = scaler.transform(input_df)
    
    # Predict
    pred_log = model.predict(input_scaled)[0]
    pred = np.expm1(pred_log)
    
    st.success(f"Prix prédit : **${pred:,.0f}**")
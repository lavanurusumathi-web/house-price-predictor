"""
api.py - Flask API for house price prediction (Render deployment)
"""
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["*"])

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

model = None
scaler = None
feature_names = None


def load_model():
    global model, scaler, feature_names
    model_path = os.path.join(MODEL_PATH, 'ridge_(l2).pkl')
    scaler_path = os.path.join(MODEL_PATH, 'scaler.pkl')
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(os.path.join(DATA_PATH, 'house_data.csv'), parse_dates=['sale_date'])
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month
    X = df.drop(['price', 'sale_date'], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()


@app.route('/')
def health():
    return jsonify({'status': 'ok', 'service': 'house-price-prediction-api'})


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input provided'}), 400

    required = ['sqft', 'bedrooms', 'bathrooms', 'year_built', 'location_score',
                'distance_to_city_miles', 'crime_rate', 'school_rating',
                'has_garage', 'has_garden', 'floors', 'sale_year']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    house_age = 2025 - int(data['year_built'])
    sale_month = 6

    row = {
        'sqft': float(data['sqft']),
        'bedrooms': int(data['bedrooms']),
        'bathrooms': float(data['bathrooms']),
        'year_built': int(data['year_built']),
        'house_age': house_age,
        'location_score': float(data['location_score']),
        'distance_to_city_miles': float(data['distance_to_city_miles']),
        'crime_rate': float(data['crime_rate']),
        'school_rating': float(data['school_rating']),
        'has_garage': int(data['has_garage']),
        'has_garden': int(data['has_garden']),
        'floors': int(data['floors']),
        'sale_year': int(data['sale_year']),
        'sale_month': sale_month
    }

    df_input = pd.DataFrame([row])
    df_input = pd.get_dummies(df_input, drop_first=True)

    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_names]

    X_scaled = scaler.transform(df_input)
    prediction = model.predict(X_scaled)[0]

    return jsonify({
        'predicted_price': round(float(prediction), 2),
        'currency': 'USD'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    load_model()
    app.run(host='0.0.0.0', port=port)

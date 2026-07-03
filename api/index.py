import os
import traceback
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

FEATURE_NAMES = [
    'sqft', 'bedrooms', 'bathrooms', 'year_built', 'house_age',
    'location_score', 'distance_to_city_miles', 'crime_rate',
    'school_rating', 'has_garage', 'has_garden', 'floors',
    'sale_year', 'sale_month'
]

_model = None
_scaler = None
_load_error = None


def _init():
    global _model, _scaler, _load_error
    _model = joblib.load(os.path.join(MODEL_DIR, 'ridge_(l2).pkl'))
    _scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))


try:
    _init()
except Exception as e:
    _load_error = traceback.format_exc()


def _response(data, status=200):
    resp = jsonify(data)
    resp.status_code = status
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp


@app.route('/api/predict', methods=['GET', 'POST', 'OPTIONS'])
def predict():
    if _load_error:
        return _response({'error': 'Model failed to load', 'detail': _load_error.split('\n')[-2]}, 500)

    if request.method == 'OPTIONS':
        return _response({})

    if request.method == 'GET':
        return _response({
            'status': 'ok',
            'service': 'house-price-prediction-api',
            'usage': 'POST with JSON: sqft, bedrooms, bathrooms, year_built, location_score, distance_to_city_miles, crime_rate, school_rating, has_garage, has_garden, floors, sale_year'
        })

    data = request.get_json(silent=True)
    if not data:
        return _response({'error': 'No JSON input provided'}, 400)

    required = ['sqft', 'bedrooms', 'bathrooms', 'year_built', 'location_score',
                'distance_to_city_miles', 'crime_rate', 'school_rating',
                'has_garage', 'has_garden', 'floors', 'sale_year']
    for field in required:
        if field not in data:
            return _response({'error': f'Missing field: {field}'}, 400)

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

    features = np.array([[row[name] for name in FEATURE_NAMES]])
    X_scaled = _scaler.transform(features)
    prediction = _model.predict(X_scaled)[0]

    return _response({
        'predicted_price': round(float(prediction), 2),
        'currency': 'USD'
    })

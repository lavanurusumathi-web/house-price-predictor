import os
import json
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

_model = None
_scaler = None
_feature_names = None


def _init():
    global _model, _scaler, _feature_names
    _model = joblib.load(os.path.join(MODEL_DIR, 'ridge_(l2).pkl'))
    _scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

    df = pd.read_csv(os.path.join(DATA_DIR, 'house_data.csv'), parse_dates=['sale_date'])
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month
    X = df.drop(['price', 'sale_date'], axis=1)
    X = pd.get_dummies(X, drop_first=True)
    _feature_names = X.columns.tolist()


def handler(request):
    if _model is None:
        _init()

    if request.method == 'GET':
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'status': 'ok', 'service': 'house-price-prediction-api'})
        }

    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': ''
        }

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': 'Invalid JSON'})
            }

        required = ['sqft', 'bedrooms', 'bathrooms', 'year_built', 'location_score',
                     'distance_to_city_miles', 'crime_rate', 'school_rating',
                     'has_garage', 'has_garden', 'floors', 'sale_year']
        for field in required:
            if field not in data:
                return {
                    'statusCode': 400,
                    'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'error': f'Missing field: {field}'})
                }

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

        for col in _feature_names:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[_feature_names]

        X_scaled = _scaler.transform(df_input)
        prediction = _model.predict(X_scaled)[0]

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'predicted_price': round(float(prediction), 2),
                'currency': 'USD'
            })
        }

    return {
        'statusCode': 405,
        'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
        'body': json.dumps({'error': 'Method not allowed'})
    }

"""
app.py - Flask web app for house price prediction
"""
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

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


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Prediction</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            color: #e0e0e0;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 40px;
            max-width: 700px;
            width: 100%;
        }
        h1 {
            text-align: center;
            font-size: 2rem;
            margin-bottom: 8px;
            background: linear-gradient(90deg, #f7971e, #ffd200);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { text-align: center; color: #999; margin-bottom: 30px; font-size: 0.95rem; }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group { display: flex; flex-direction: column; }
        .form-group.full { grid-column: 1 / -1; }
        label { font-size: 0.85rem; color: #bbb; margin-bottom: 6px; font-weight: 500; }
        input, select {
            padding: 12px 14px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.15);
            background: rgba(255,255,255,0.06);
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }
        input:focus, select:focus { border-color: #f7971e; }
        select option { background: #302b63; color: #fff; }
        button {
            grid-column: 1 / -1;
            padding: 14px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(90deg, #f7971e, #ffd200);
            color: #1a1a2e;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            margin-top: 10px;
            transition: transform 0.15s, box-shadow 0.15s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(247,151,30,0.4); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }
        .result {
            margin-top: 30px;
            padding: 25px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;
            display: none;
        }
        .result.show { display: block; }
        .predicted-price {
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .price-label { text-align: center; color: #999; font-size: 0.9rem; margin-bottom: 20px; }
        .details { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .detail-item {
            background: rgba(255,255,255,0.04);
            padding: 12px 16px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
        }
        .detail-item .label { color: #888; }
        .detail-item .value { color: #fff; font-weight: 600; }
        .error-msg {
            margin-top: 30px;
            padding: 16px;
            background: rgba(220,53,69,0.15);
            border: 1px solid rgba(220,53,69,0.3);
            border-radius: 10px;
            color: #f8516a;
            text-align: center;
            display: none;
        }
        .error-msg.show { display: block; }
        .spinner { display: none; text-align: center; padding: 10px; }
        .spinner.show { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>House Price Predictor</h1>
        <p class="subtitle">Enter property details to estimate its market value</p>

        <form id="predict-form">
            <div class="form-grid">
                <div class="form-group">
                    <label for="sqft">Square Footage</label>
                    <input type="number" id="sqft" placeholder="e.g. 1800" min="600" max="5000" required>
                </div>
                <div class="form-group">
                    <label for="bedrooms">Bedrooms</label>
                    <select id="bedrooms" required>
                        <option value="">Select</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3" selected>3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                        <option value="6">6</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="bathrooms">Bathrooms</label>
                    <select id="bathrooms" required>
                        <option value="">Select</option>
                        <option value="1">1.0</option>
                        <option value="1.5">1.5</option>
                        <option value="2">2.0</option>
                        <option value="2.5">2.5</option>
                        <option value="3">3.0</option>
                        <option value="3.5">3.5</option>
                        <option value="4">4.0</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="year_built">Year Built</label>
                    <input type="number" id="year_built" placeholder="e.g. 2005" min="1950" max="2025" value="2005" required>
                </div>
                <div class="form-group">
                    <label for="location_score">Location Score (1-10)</label>
                    <input type="number" id="location_score" placeholder="e.g. 7.5" min="1" max="10" step="0.1" value="6" required>
                </div>
                <div class="form-group">
                    <label for="distance_to_city">Distance to City (miles)</label>
                    <input type="number" id="distance_to_city" placeholder="e.g. 5.0" min="0.5" max="40" step="0.1" value="8" required>
                </div>
                <div class="form-group">
                    <label for="crime_rate">Crime Rate (0.5-10)</label>
                    <input type="number" id="crime_rate" placeholder="e.g. 3.0" min="0.5" max="10" step="0.1" value="4" required>
                </div>
                <div class="form-group">
                    <label for="school_rating">School Rating (1-10)</label>
                    <input type="number" id="school_rating" placeholder="e.g. 7.0" min="1" max="10" step="0.1" value="6" required>
                </div>
                <div class="form-group">
                    <label for="has_garage">Garage</label>
                    <select id="has_garage" required>
                        <option value="1" selected>Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="has_garden">Garden</label>
                    <select id="has_garden" required>
                        <option value="1" selected>Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="floors">Floors</label>
                    <select id="floors" required>
                        <option value="1">1</option>
                        <option value="2" selected>2</option>
                        <option value="3">3</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="sale_year">Sale Year</label>
                    <input type="number" id="sale_year" min="2020" max="2035" value="2025" required>
                </div>
                <button type="submit" id="submit-btn">Predict Price</button>
            </div>
        </form>

        <div class="spinner" id="spinner">Calculating...</div>
        <div class="error-msg" id="error"></div>

        <div class="result" id="result">
            <div class="predicted-price" id="predicted-price">--</div>
            <div class="price-label">Estimated Market Value (USD)</div>
            <div class="details" id="details"></div>
        </div>
    </div>

    <script>
        const form = document.getElementById('predict-form');
        const resultDiv = document.getElementById('result');
        const errorDiv = document.getElementById('error');
        const spinner = document.getElementById('spinner');
        const submitBtn = document.getElementById('submit-btn');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            resultDiv.classList.remove('show');
            errorDiv.classList.remove('show');
            spinner.classList.add('show');
            submitBtn.disabled = true;

            const data = {
                sqft: parseFloat(document.getElementById('sqft').value),
                bedrooms: parseInt(document.getElementById('bedrooms').value),
                bathrooms: parseFloat(document.getElementById('bathrooms').value),
                year_built: parseInt(document.getElementById('year_built').value),
                location_score: parseFloat(document.getElementById('location_score').value),
                distance_to_city_miles: parseFloat(document.getElementById('distance_to_city').value),
                crime_rate: parseFloat(document.getElementById('crime_rate').value),
                school_rating: parseFloat(document.getElementById('school_rating').value),
                has_garage: parseInt(document.getElementById('has_garage').value),
                has_garden: parseInt(document.getElementById('has_garden').value),
                floors: parseInt(document.getElementById('floors').value),
                sale_year: parseInt(document.getElementById('sale_year').value)
            };

            try {
                const resp = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const json = await resp.json();

                if (!resp.ok) {
                    throw new Error(json.error || 'Prediction failed');
                }

                document.getElementById('predicted-price').textContent =
                    '$' + json.predicted_price.toLocaleString('en-US');

                const detailsDiv = document.getElementById('details');
                detailsDiv.innerHTML = '';
                const details = [
                    ['Square Footage', data.sqft + ' sqft'],
                    ['Bedrooms', data.bedrooms],
                    ['Bathrooms', data.bathrooms],
                    ['Year Built', data.year_built],
                    ['Location Score', data.location_score + '/10'],
                    ['Distance to City', data.distance_to_city_miles + ' mi'],
                    ['Crime Rate', data.crime_rate + '/10'],
                    ['School Rating', data.school_rating + '/10'],
                    ['Garage', data.has_garage ? 'Yes' : 'No'],
                    ['Garden', data.has_garden ? 'Yes' : 'No'],
                    ['Floors', data.floors],
                    ['Sale Year', data.sale_year]
                ];
                details.forEach(([label, value]) => {
                    detailsDiv.innerHTML +=
                        `<div class="detail-item"><span class="label">${label}</span><span class="value">${value}</span></div>`;
                });

                resultDiv.classList.add('show');
            } catch (err) {
                errorDiv.textContent = err.message;
                errorDiv.classList.add('show');
            } finally {
                spinner.classList.remove('show');
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


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
    print("Loading model...")
    load_model()
    print("Model loaded. Starting server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

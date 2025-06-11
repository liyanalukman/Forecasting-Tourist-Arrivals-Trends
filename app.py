from flask import Flask, redirect, url_for, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pycountry
import io
import base64

# Load dataset
df = pd.read_csv('arrivals_soe.csv')
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.strftime('%b %Y')
last_data_date = df['date'].max()

# Generate 12 future months from the last data point
months = pd.date_range(start=last_data_date + pd.DateOffset(months=1), periods=12, freq='MS').strftime('%b %Y').tolist()

# Helper to convert 3-letter country code to full name
def get_country_name(code):
    try:
        return pycountry.countries.get(alpha_3=code).name
    except:
        return None

# Map country codes to full names
country_code_to_name = {}
for code in df['country'].unique():
    name = get_country_name(code)
    if name:
        country_code_to_name[code] = name
    else:
        country_code_to_name[code] = code  # fallback to code

# Monthly total arrivals
monthly_totals = df.groupby('month')['arrivals'].sum().to_dict()

# Country proportions by month
proportion_by_month = {}
for month, group in df.groupby('month'):
    total = monthly_totals.get(month, 1)
    proportions = (group.set_index('country')['arrivals'] / total).to_dict()
    proportion_by_month[month] = proportions

# State (SOE) proportions by month
state_proportion_by_month = {}
for month, group in df.groupby('month'):
    total = monthly_totals.get(month, 1)
    proportions = (group.groupby('soe')['arrivals'].sum() / total).to_dict()
    state_proportion_by_month[month] = proportions

# Initialize Flask app
app = Flask(__name__)

# Load SARIMA model
model = joblib.load('sarima_model.pkl')

# ----------------- ROUTING --------------------------------#
@app.route('/')
def home():
    return redirect(url_for('about'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict/monthly')
def monthly_predict_page():
    return render_template('monthly_predict.html')

@app.route('/predict/state')
def state_predict_page():
    return render_template('state_predict.html')

@app.route('/predict/country')
def country_predict_page():
    return render_template('country_predict.html')

# -------------------- GET FROM DATASET ------------------#
@app.route('/months')
def get_months():
    return jsonify(months)

@app.route('/states')
def get_states():
    states = sorted(df['soe'].dropna().unique().tolist())
    return jsonify(states)

@app.route('/countries')
def get_countries():
    base_month = last_data_date.strftime('%b %Y')
    country_codes = proportion_by_month.get(base_month, {})
    country_names = [country_code_to_name.get(code, code) for code in country_codes]
    country_names = list(filter(None, country_names))
    return jsonify(country_names)

# ----------------- SEND DATA TO HTML -----------------#
@app.route('/predict/monthly', methods=['POST'])
def predict_monthly():
    selected_month = request.get_json().get('selected_month')
    if not selected_month:
        return jsonify({'error': 'No month selected'})

    pred_date = pd.to_datetime(selected_month, format='%b %Y')
    months_diff = (pred_date.year - last_data_date.year) * 12 + (pred_date.month - last_data_date.month)
    base_str = last_data_date.strftime('%b %Y')

    total_arrivals_year = df.groupby('country')['arrivals'].sum()
    total_all = total_arrivals_year.sum()
    country_props = (total_arrivals_year / total_all).to_dict()

    state_props = state_proportion_by_month.get(base_str, {})

    if months_diff > 0:
        forecast = model.forecast(steps=months_diff)
        predicted_total = forecast[-1]
    else:
        predicted_total = model.predict()[months_diff]

    predicted_total = max(predicted_total, 0)

    top_states = sorted(
        ((state, int(predicted_total * p)) for state, p in state_props.items()),
        key=lambda x: x[1], reverse=True)[:5]

    top_countries = sorted(
        ((country_code_to_name.get(c, c), int(predicted_total * p)) for c, p in country_props.items()),
        key=lambda x: x[1], reverse=True)[:5]

    return jsonify({
        'month': selected_month,
        'prediction': int(predicted_total),
        'top_countries': top_countries,
        'top_states': top_states
    })

@app.route('/predict/state_trend', methods=['POST'])
def predict_state_trend():
    selected_state = request.get_json().get('state')
    if not selected_state:
        return jsonify({'error': 'No state selected'})

    forecast = model.forecast(steps=12)
    forecast = np.maximum(forecast, 0)

    base_month = last_data_date.strftime('%b %Y')
    state_props = state_proportion_by_month.get(base_month, {})
    state_prop = state_props.get(selected_state, 0)

    state_forecast = [int(total * state_prop) for total in forecast]

    return jsonify({
        'months': months,
        'arrivals': state_forecast
    })

@app.route('/predict/country_trend', methods=['POST'])
def predict_country_trend():
    selected_country = request.get_json().get('country')
    if not selected_country:
        return jsonify({'error': 'No country selected'})

    forecast = model.forecast(steps=12)
    forecast = np.maximum(forecast, 0)

    base_month = last_data_date.strftime('%b %Y')

    country_code = None
    for code in proportion_by_month.get(base_month, {}):
        if country_code_to_name.get(code) == selected_country:
            country_code = code
            break

    if not country_code:
        return jsonify({'error': 'Country not found'})

    country_props = proportion_by_month.get(base_month, {})
    country_prop = country_props.get(country_code, 0)

    country_forecast = [int(total * country_prop) for total in forecast]

    return jsonify({
        'months': months,
        'arrivals': country_forecast
    })

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, redirect, url_for, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Last date in your dataset (ensure this is correct based on the CSV)
last_data_date = pd.to_datetime('2024-10-01')

# Generate 12 future months from the last data point
months = pd.date_range(start=last_data_date + pd.DateOffset(months=1), periods=12, freq='MS').strftime('%b %Y').tolist()

# Load dataset
df = pd.read_csv('arrivals_soe.csv')
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.strftime('%b %Y')

# Map country codes to names (if column exists)
if 'country_name' in df.columns:
    country_code_to_name = dict(zip(df['country'], df['country_name']))
else:
    country_code_to_name = {code: code for code in df['country'].unique()}

# Monthly total arrivals
monthly_totals = df.groupby('month')['arrivals'].sum().to_dict()

# Country proportions by month
proportion_by_month = {}
for month, group in df.groupby('month'):
    total = monthly_totals.get(month, 1)  # avoid division by zero
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

@app.route('/')
def home():
    return redirect(url_for('about'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict_page():
    return render_template('index.html')

@app.route('/months')
def get_months():
    return jsonify(months)

@app.route('/predict', methods=['POST'])
def predict():
    selected_month = request.get_json().get('selected_month')
    if not selected_month:
        return jsonify({'error': 'No month selected'})

    # Convert selected month to datetime
    pred_date = pd.to_datetime(selected_month, format='%b %Y')
    last_date = last_data_date

    # Find how many months ahead the prediction is
    months_diff = (pred_date.year - last_date.year) * 12 + (pred_date.month - last_date.month)

    # Use base proportions from last known data
    base_str = last_date.strftime('%b %Y')

    # Proportion of each country over the full year
    total_arrivals_year = df.groupby('country')['arrivals'].sum()
    total_all = total_arrivals_year.sum()
    country_props = (total_arrivals_year / total_all).to_dict()

    #country_props = proportion_by_month.get(base_str, {})
    state_props = state_proportion_by_month.get(base_str, {})

    # Predict total arrivals
    if months_diff > 0:
        forecast = model.forecast(steps=months_diff)
        predicted_total = forecast[-1]
    else:
        predicted_total = model.predict()[months_diff]  # only valid for in-sample

    predicted_total = max(predicted_total, 0)  # ensure non-negative

    # Top 5 states
    top_states = sorted(
        ((state, int(predicted_total * p)) for state, p in state_props.items()),
        key=lambda x: x[1], reverse=True)[:5]

    # Top 5 countries
    top_countries = sorted(
        ((country_code_to_name.get(c, c), int(predicted_total * p)) for c, p in country_props.items()),
        key=lambda x: x[1], reverse=True)[:5]

    # Debug (optional, remove in production)
    print("Selected month:", selected_month)
    print("Months ahead:", months_diff)
    print("Predicted total:", predicted_total)
    print("Top countries:", top_countries)
    print("Top states:", top_states)

    return jsonify({
        'month': selected_month,
        'prediction': int(predicted_total),
        'top_countries': top_countries,
        'top_states': top_states
    })

if __name__ == '__main__':
    app.run(debug=True)

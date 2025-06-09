from flask import Flask, redirect, url_for, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

last_data_date = pd.to_datetime('2024-10-01')  # last month in your dataset
months = pd.date_range(start=last_data_date + pd.DateOffset(months=1), periods=12, freq='MS').strftime('%b %Y').tolist()

df = pd.read_csv('arrivals_soe.csv')
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.strftime('%b %Y')

if 'country_name' in df.columns:
    country_code_to_name = dict(zip(df['country'], df['country_name']))
else:
    # If no full country names, just map code to code
    country_code_to_name = {code: code for code in df['country'].unique()}

monthly_totals = df.groupby('month')['arrivals'].sum().to_dict()

proportion_by_month = {}

for month, group in df.groupby('month'):
    total = monthly_totals.get(month, 1)  # avoid div by zero
    proportions = (group.set_index('country')['arrivals'] / total).to_dict()
    proportion_by_month[month] = proportions

app = Flask(__name__)

# Load the SARIMA model
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

    pred_date = pd.to_datetime(selected_month, format='%b %Y')
    last_date = pd.to_datetime('2024-10-01')

    base_month = last_date if pred_date > last_date else pred_date
    base_str = base_month.strftime('%b %Y')
    month_data = df[df['month'] == base_str]
    if month_data.empty:
        return jsonify({'error': 'No data available for the base month'})

    proportions = (month_data.groupby('country')['arrivals'].sum() / month_data['arrivals'].sum()).to_dict()
    months_diff = (pred_date.year - last_date.year) * 12 + (pred_date.month - last_date.month)

    # Use forecast for future months, or direct predict for known months
    if months_diff > 0:
        # Predict n steps ahead
        predicted_total = model.forecast(steps=months_diff)[-1]
    else:
        # If months_diff is 0 or negative, get fitted value or actual
        predicted_total = model.predict()[months_diff]

    top_5 = sorted(
        ((country_code_to_name.get(c, c), int(predicted_total * p)) for c, p in proportions.items()), 
        key=lambda x: x[1], reverse=True
    )[:5]

    return jsonify({
        'month': selected_month,
        'prediction': int(predicted_total),
        'top_countries': top_5
    })

if __name__ == '__main__':
    app.run(debug=True)

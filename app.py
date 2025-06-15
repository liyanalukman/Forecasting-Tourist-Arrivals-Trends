from flask import Flask, redirect, url_for, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pycountry
import io
import base64
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load dataset
try:
    logger.info("Loading dataset from arrivals_soe.csv")
    df = pd.read_csv('arrivals_soe.csv')
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Sample data:\n{df.head()}")
    
    # Validate required columns
    required_columns = ['date', 'country', 'soe', 'arrivals']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Check for missing values
    missing_values = df[required_columns].isnull().sum()
    if missing_values.any():
        logger.warning(f"Missing values found:\n{missing_values}")
    
    df['month'] = df['date'].dt.strftime('%b %Y')
    last_data_date = df['date'].max()
    logger.info(f"Last data date: {last_data_date}")
    
    # Generate 12 future months from the last data point
    months = pd.date_range(start=last_data_date + pd.DateOffset(months=1), periods=12, freq='MS').strftime('%b %Y').tolist()
    logger.info(f"Generated future months: {months}")
    
except Exception as e:
    logger.error(f"Error loading or processing dataset: {str(e)}", exc_info=True)
    raise

# Helper to convert 3-letter country code to full name
def get_country_name(code):
    try:
        return pycountry.countries.get(alpha_3=code).name
    except:
        return None

# Map country codes to full names
try:
    logger.info("Mapping country codes to names")
    country_code_to_name = {}
    for code in df['country'].unique():
        name = get_country_name(code)
        if name:
            country_code_to_name[code] = name
        else:
            country_code_to_name[code] = code  # fallback to code
    logger.info(f"Country mapping complete. Total countries: {len(country_code_to_name)}")
    logger.debug(f"Country mapping sample: {dict(list(country_code_to_name.items())[:5])}")
except Exception as e:
    logger.error(f"Error mapping country codes: {str(e)}", exc_info=True)
    raise

# Monthly total arrivals
try:
    logger.info("Calculating monthly totals")
    monthly_totals = df.groupby('month')['arrivals'].sum().to_dict()
    logger.info(f"Monthly totals calculated. Total months: {len(monthly_totals)}")
    logger.debug(f"Monthly totals sample: {dict(list(monthly_totals.items())[:5])}")
except Exception as e:
    logger.error(f"Error calculating monthly totals: {str(e)}", exc_info=True)
    raise

# Country proportions by month
try:
    logger.info("Calculating country proportions by month")
    proportion_by_month = {}
    for month, group in df.groupby('month'):
        total = monthly_totals.get(month, 1)
        proportions = (group.set_index('country')['arrivals'] / total).to_dict()
        proportion_by_month[month] = proportions
    logger.info(f"Country proportions calculated. Total months: {len(proportion_by_month)}")
except Exception as e:
    logger.error(f"Error calculating country proportions: {str(e)}", exc_info=True)
    raise

# State (SOE) proportions by month
try:
    logger.info("Calculating state proportions by month")
    state_proportion_by_month = {}
    for month, group in df.groupby('month'):
        total = monthly_totals.get(month, 1)
        proportions = (group.groupby('soe')['arrivals'].sum() / total).to_dict()
        state_proportion_by_month[month] = proportions
    logger.info(f"State proportions calculated. Total months: {len(state_proportion_by_month)}")
except Exception as e:
    logger.error(f"Error calculating state proportions: {str(e)}", exc_info=True)
    raise

# Initialize Flask app
app = Flask(__name__)

# Load SARIMA model
try:
    logger.info("Loading SARIMA model")
    model = joblib.load('sarima_model.pkl')
    logger.info("SARIMA model loaded successfully")
except Exception as e:
    logger.error(f"Error loading SARIMA model: {str(e)}", exc_info=True)
    raise

# ----------------- ROUTING --------------------------------#
@app.route('/')
def home():
    return redirect(url_for('about'))

@app.route('/about')
def about():
    # Predict next month's total arrivals
    next_month_forecast = model.forecast(steps=1)
    predicted_total = int(max(next_month_forecast[0], 0))  # Ensure it's non-negative

    # Get latest month's state data
    base_str = last_data_date.strftime('%b %Y')
    state_props = state_proportion_by_month.get(base_str, {})
    state_totals = {state: int(predicted_total * prop) for state, prop in state_props.items()}
    
    # Sort by arrivals descending
    sorted_states = sorted(state_totals.items(), key=lambda x: x[1], reverse=True)

    # Send data to template
    return render_template(
        'about.html',
        predicted_total=predicted_total,
        state_arrivals=sorted_states
    )


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

@app.route('/api/sneak_peek')
def sneak_peek():
    try:
        logger.info("Generating sneak peek predictions")
        # Generate predictions for next 6 months
        forecast = model.forecast(steps=6)
        forecast = np.maximum(forecast, 0)  # Ensure non-negative values
        
        # Create list of predictions with months
        predictions = []
        for i, pred in enumerate(forecast):
            pred_date = last_data_date + pd.DateOffset(months=i+1)
            predictions.append({
                'month': pred_date.strftime('%b %Y'),
                'yhat': float(pred)  # Convert numpy float to Python float
            })
        
        logger.info(f"Generated {len(predictions)} sneak peek predictions")
        return jsonify(predictions)
    except Exception as e:
        logger.error(f"Error generating sneak peek: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

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

        # Get total arrivals for each state
    state_totals = df.groupby('soe')['arrivals'].sum()
    total_arrivals = state_totals.sum()

    average_state_props = (state_totals / total_arrivals).to_dict()

    # Get selected state proportion
    state_prop = average_state_props.get(selected_state, 0)

    # Forecast future arrivals
    forecast = model.forecast(steps=12)
    forecast = np.maximum(forecast, 0)
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

    country_totals = df.groupby('country')['arrivals'].sum()
    total_arrivals = country_totals.sum()
    average_props = (country_totals / total_arrivals).to_dict()

    # Match country name to code
    country_code = None
    for code, name in country_code_to_name.items():
        if name == selected_country:
            country_code = code
            break

    if not country_code:
        return jsonify({'error': 'Country not found'})

    country_prop = average_props.get(country_code, 0)

    country_forecast = [int(total * country_prop) for total in forecast]

    return jsonify({
        'months': months,
        'arrivals': country_forecast
    })

# Add new routes for analytics
@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/api/analytics/yoy')
def yoy_comparison():
    try:
        logger.info("Generating year-over-year comparison")
        # Calculate year-over-year comparison
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        yearly_data = df.groupby(['year', 'month'])['arrivals'].sum().reset_index()
        
        # Create traces for each year
        traces = []
        years = sorted(yearly_data['year'].unique())
        
        for year in years:
            year_data = yearly_data[yearly_data['year'] == year]
            # Ensure data is properly sorted by month
            year_data = year_data.sort_values('month')
            
            # Convert to Python native types and handle NaN values
            x_values = year_data['month'].tolist()
            y_values = [None if pd.isna(val) else round(float(val)) for val in year_data['arrivals']]
            
            traces.append({
                'x': x_values,
                'y': y_values,
                'name': str(year),
                'type': 'scatter',
                'mode': 'lines+markers'
            })
        
        layout = {
            'title': 'Year-over-Year Comparison',
            'xaxis': {
                'title': 'Month',
                'ticktext': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'tickvals': list(range(1, 13))
            },
            'yaxis': {
                'title': 'Tourist Arrivals'
            },
            'hovermode': 'x unified'
        }
        
        logger.info("Year-over-year comparison created successfully")
        return jsonify({'data': traces, 'layout': layout})
    except Exception as e:
        logger.error(f"Error generating year-over-year comparison: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/seasonal')
def seasonal_decomposition():
    try:
        logger.info("Generating seasonal decomposition")
        # Prepare time series data
        monthly_data = df.groupby('date')['arrivals'].sum()
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(monthly_data, period=12)
        
        # Create subplots
        fig = make_subplots(rows=4, cols=1, 
                           subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
        
        # Helper function to convert data to JSON-safe format
        def convert_to_json_safe(data):
            return [None if pd.isna(val) else float(val) for val in data]
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=monthly_data.index.strftime('%Y-%m-%d').tolist(),
            y=convert_to_json_safe(monthly_data.values),
            name='Original'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=decomposition.trend.index.strftime('%Y-%m-%d').tolist(),
            y=convert_to_json_safe(decomposition.trend.values),
            name='Trend'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=decomposition.seasonal.index.strftime('%Y-%m-%d').tolist(),
            y=convert_to_json_safe(decomposition.seasonal.values),
            name='Seasonal'
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=decomposition.resid.index.strftime('%Y-%m-%d').tolist(),
            y=convert_to_json_safe(decomposition.resid.values),
            name='Residual'
        ), row=4, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Seasonal Decomposition of Tourist Arrivals"
        )
        
        logger.info("Seasonal decomposition created successfully")
        return jsonify(fig.to_dict())
    except Exception as e:
        logger.error(f"Error generating seasonal decomposition: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/moving-avg')
def moving_averages():
    try:
        logger.info("Generating moving averages")
        # Calculate moving averages
        monthly_data = df.groupby('date')['arrivals'].sum()
        ma_3 = monthly_data.rolling(window=3).mean()
        ma_6 = monthly_data.rolling(window=6).mean()
        ma_12 = monthly_data.rolling(window=12).mean()
        
        # Helper function to convert data to JSON-safe format
        def convert_to_json_safe(data):
            return [None if pd.isna(val) else float(val) for val in data]
        
        # Create traces
        traces = [
            {
                'x': monthly_data.index.strftime('%Y-%m-%d').tolist(),
                'y': convert_to_json_safe(monthly_data.values),
                'name': 'Original',
                'type': 'scatter',
                'mode': 'lines'
            },
            {
                'x': ma_3.index.strftime('%Y-%m-%d').tolist(),
                'y': convert_to_json_safe(ma_3.values),
                'name': '3-Month MA',
                'type': 'scatter',
                'mode': 'lines'
            },
            {
                'x': ma_6.index.strftime('%Y-%m-%d').tolist(),
                'y': convert_to_json_safe(ma_6.values),
                'name': '6-Month MA',
                'type': 'scatter',
                'mode': 'lines'
            },
            {
                'x': ma_12.index.strftime('%Y-%m-%d').tolist(),
                'y': convert_to_json_safe(ma_12.values),
                'name': '12-Month MA',
                'type': 'scatter',
                'mode': 'lines'
            }
        ]
        
        layout = {
            'title': 'Moving Averages Analysis',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Tourist Arrivals'},
            'hovermode': 'x unified'
        }
        
        logger.info("Moving averages created successfully")
        return jsonify({'data': traces, 'layout': layout})
    except Exception as e:
        logger.error(f"Error generating moving averages: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/peak-season')
def peak_season():
    try:
        logger.info("Generating peak season analysis")
        # Calculate monthly averages
        monthly_avg = df.groupby(df['date'].dt.month)['arrivals'].mean()
        
        # Convert to JSON-safe format
        x_values = monthly_avg.index.tolist()
        y_values = [None if pd.isna(val) else float(val) for val in monthly_avg.values]
        text_values = [None if pd.isna(val) else str(round(val)) for val in monthly_avg.values]
        
        # Create bar chart
        trace = {
            'x': x_values,
            'y': y_values,
            'text': text_values,
            'type': 'bar',
            'textposition': 'auto'
        }
        
        layout = {
            'title': 'Average Monthly Tourist Arrivals',
            'xaxis': {
                'title': 'Month',
                'ticktext': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'tickvals': list(range(1, 13))
            },
            'yaxis': {'title': 'Average Tourist Arrivals'}
        }
        
        logger.info("Peak season analysis created successfully")
        return jsonify({'data': [trace], 'layout': layout})
    except Exception as e:
        logger.error(f"Error generating peak season analysis: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/growth-rate')
def growth_rate():
    try:
        logger.info("Generating growth rate analysis")
        # Calculate monthly growth rates
        monthly_data = df.groupby('date')['arrivals'].sum()
        growth_rates = monthly_data.pct_change() * 100
        
        # Helper function to convert data to JSON-safe format
        def convert_to_json_safe(data):
            return [None if pd.isna(val) else float(val) for val in data]
        
        # Create traces
        traces = [
            {
                'x': monthly_data.index.strftime('%Y-%m-%d').tolist(),
                'y': convert_to_json_safe(monthly_data.values),
                'name': 'Arrivals',
                'type': 'scatter',
                'yaxis': 'y1'
            },
            {
                'x': growth_rates.index.strftime('%Y-%m-%d').tolist(),
                'y': convert_to_json_safe(growth_rates.values),
                'name': 'Growth Rate (%)',
                'type': 'scatter',
                'yaxis': 'y2'
            }
        ]
        
        layout = {
            'title': 'Tourist Arrivals and Growth Rate',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Tourist Arrivals', 'side': 'left'},
            'yaxis2': {
                'title': 'Growth Rate (%)',
                'side': 'right',
                'overlaying': 'y',
                'showgrid': False
            },
            'hovermode': 'x unified'
        }
        
        logger.info("Growth rate analysis created successfully")
        return jsonify({'data': traces, 'layout': layout})
    except Exception as e:
        logger.error(f"Error generating growth rate analysis: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/interactive_viz')
def interactive_viz():
    try:
        logger.info("Rendering interactive_viz template")
        return render_template('interactive_viz.html')
    except Exception as e:
        logger.error(f"Error rendering interactive_viz template: {str(e)}", exc_info=True)
        return str(e), 500

@app.route('/api/viz/state-map')
def state_map():
    try:
        logger.info("Generating state map visualization")
        # Calculate total arrivals by state
        state_data = df.groupby('soe')['arrivals'].sum().reset_index()
        logger.info(f"State data calculated. Shape: {state_data.shape}")
        logger.debug(f"State data sample: {state_data.head()}")
        
        if state_data.empty:
            raise ValueError("No state data available")
        
        # Create bar chart for state distribution
        fig = go.Figure(data=[
            go.Bar(
                x=state_data['soe'].tolist(),
                y=state_data['arrivals'].tolist(),
                text=state_data['arrivals'].round().tolist(),
                textposition='auto',
                hovertemplate='State: %{x}<br>Arrivals: %{y:,.0f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Tourist Distribution Across States',
            xaxis_title='State',
            yaxis_title='Total Arrivals',
            height=500,
            showlegend=False
        )
        
        logger.info("State map visualization created successfully")
        return jsonify(fig.to_dict())
    except Exception as e:
        logger.error(f"Error generating state map: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/viz/drill-down')
def drill_down():
    try:
        logger.info("Generating drill-down chart")
        # Get top 10 countries by total arrivals
        country_data = df.groupby('country')['arrivals'].sum()
        country_data = country_data.sort_values(ascending=False).head(10)
        logger.info(f"Top 10 countries data calculated. Shape: {country_data.shape}")
        logger.debug(f"Country data sample:\n{country_data.head()}")
        
        if country_data.empty:
            raise ValueError("No country data available")
        
        # Convert country codes to names
        country_names = [country_code_to_name.get(code, code) for code in country_data.index]
        logger.info(f"Country names converted: {country_names}")
        
        # Create bar chart with drill-down capability
        fig = go.Figure(data=[
            go.Bar(
                x=country_names,
                y=country_data.values.tolist(),
                text=country_data.values.round().tolist(),
                textposition='auto',
                hovertemplate='Country: %{x}<br>Arrivals: %{y:,.0f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Top 10 Countries by Tourist Arrivals',
            xaxis_title='Country',
            yaxis_title='Total Arrivals',
            height=500,
            showlegend=False
        )
        
        logger.info("Drill-down chart created successfully")
        return jsonify(fig.to_dict())
    except Exception as e:
        logger.error(f"Error generating drill-down chart: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/viz/interactive-line')
def interactive_line():
    try:
        logger.info("Generating interactive line chart")
        # Calculate monthly totals
        monthly_data = df.groupby('date')['arrivals'].sum().reset_index()
        logger.info(f"Monthly data calculated. Shape: {monthly_data.shape}")
        logger.debug(f"Monthly data sample:\n{monthly_data.head()}")
        
        if monthly_data.empty:
            raise ValueError("No monthly data available")
        
        # Create interactive line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            y=monthly_data['arrivals'].tolist(),
            mode='lines+markers',
            name='Tourist Arrivals',
            hovertemplate='Date: %{x}<br>Arrivals: %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Tourist Arrivals Trend',
            xaxis_title='Date',
            yaxis_title='Arrivals',
            height=500,
            hovermode='x unified',
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        
        logger.info("Interactive line chart created successfully")
        return jsonify(fig.to_dict())
    except Exception as e:
        logger.error(f"Error generating interactive line chart: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/viz/market-share')
def market_share():
    try:
        logger.info("Generating market share chart")
        # Calculate market share by country
        country_data = df.groupby('country')['arrivals'].sum()
        total_arrivals = country_data.sum()
        market_share = (country_data / total_arrivals * 100).round(1)
        logger.info(f"Market share calculated. Total countries: {len(market_share)}")
        logger.debug(f"Market share sample:\n{market_share.head()}")
        
        if market_share.empty:
            raise ValueError("No market share data available")
        
        # Get top 5 countries and combine others
        top_5 = market_share.nlargest(5)
        others = pd.Series({'Others': market_share.nsmallest(len(market_share) - 5).sum()})
        final_data = pd.concat([top_5, others])
        logger.info(f"Top 5 countries and others calculated. Total categories: {len(final_data)}")
        logger.debug(f"Final data sample:\n{final_data}")
        
        # Convert country codes to names
        country_names = [country_code_to_name.get(code, code) for code in final_data.index]
        logger.info(f"Country names converted: {country_names}")
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=country_names,
            values=final_data.values.tolist(),
            textinfo='label+percent',
            insidetextorientation='radial',
            hovertemplate='Country: %{label}<br>Market Share: %{value:.1f}%<extra></extra>'
        )])
        
        fig.update_layout(
            title='Tourist Market Share by Country',
            height=500,
            showlegend=True
        )
        
        logger.info("Market share chart created successfully")
        return jsonify(fig.to_dict())
    except Exception as e:
        logger.error(f"Error generating market share chart: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

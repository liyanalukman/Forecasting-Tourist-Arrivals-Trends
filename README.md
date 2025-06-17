# Forecasting Tourist Arrivals Trends in Malaysia 🇲🇾

This project aims to analyze, model, and forecast monthly tourist arrivals to Malaysia using machine learning and time series models. The final output is an interactive web application built with Flask and HTML to support tourism planning and policy decisions.

---

## 📊 Project Overview

- **Data Source**: Malaysian Immigration System (MyIMMs) via [data.gov.my](https://data.gov.my/data-catalogue/arrivals_soe)
- **Time Period**: January 2020 – October 2024
- **Objective**: To forecast monthly tourist arrivals segmented by country and provide a user-friendly interface for exploration.

---

## 🧪 Reproducible Research Workflow

This project is structured to be fully transparent and reproducible:

- All analysis done in **Python (Google Colab)**.
- Modeling conducted with **XGBoost**, **LightGBM**, and **SARIMA**.
- All notebooks and scripts are executable and annotated.
- Data preprocessing, EDA, and modeling steps are documented clearly.
- Final results are hosted in this repository and integrated into a web app.

---

## 🧠 Models Used

| Model      | Description                                              |
|------------|----------------------------------------------------------|
| XGBoost    | Powerful gradient boosting model used for regression     |
| LightGBM   | Fast, memory-efficient model with good performance       |
| SARIMA     | Time series model for capturing seasonal patterns        |

---

## 🔮 Features of the Web App (Flask + HTML)

- Forecasts tourist arrivals based on trained models.
- Interactive visualizations using Plotly (trend, seasonality, and more).
- Allows country-specific filtering (e.g., Singapore, China, etc.).
- Historical and predicted data displayed with confidence intervals.
- Responsive UI for both technical and non-technical users.

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/liyanalukman/Forecasting-Tourist-Arrivals-Trends.git
cd Forecasting-Tourist-Arrivals-Trends
```
### 2. Create Virtual Environment 
```bash
python -m venv env
env\Scripts\activate //for Windows
source env/bin/activate // for Linux/macOS
```
### 3. Run the Flask Application
```bash
python app.py
```

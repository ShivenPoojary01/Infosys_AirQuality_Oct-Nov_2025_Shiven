import os
import time
import zipfile  # To zip up saved models
import joblib   # For saving/loading ML models
import warnings
warnings.filterwarnings("ignore") # Hide warnings from ML libs

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Optional ML libs (graceful fallback)
# We try to import each one. If it fails, we set a flag so the app
# doesn't crash and just disables that model.
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Models folder
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True) # Create the 'models' folder if it doesn't exist

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="AirAware — Air Quality Dashboard", layout="wide", initial_sidebar_state="expanded")

# --------------------------
# Material-style CSS (Light only)
# --------------------------
# This is a big block of custom CSS to make the app look like a
# clean, modern "Material Design" dashboard.
css = """
<style>
/* Page background and base */
:root{
  --bg: #f5f7fb;
  --card: #ffffff;
  --muted: #6b7280;
  --primary: #2f59ff; /* primary blue */
  --accent: #6c8cff;
  --success: #10b981;
  --warn: #f59e0b;
  --danger: #ef4444;
  --surface-shadow: 0 6px 18px rgba(40,41,61,0.06);
}
.css-1d391kg { padding: 0 !important; }

/* App background */
.stApp {
  background: linear-gradient(180deg, var(--bg) 0%, #eef3fb 100%);
  color: #0f172a;
  font-family: Inter, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff, #fbfdff);
  border-right: 1px solid rgba(15,23,42,0.04);
  padding: 18px;
  box-shadow: none;
}
[data-testid="stSidebar"] .css-1d392kg { padding-top: 8px; }

/* Header */
.header {
  background: linear-gradient(90deg, rgba(47,89,255,0.95) 0%, rgba(108,140,255,0.95) 100%);
  color: white;
  padding: 22px;
  border-radius: 10px;
  margin-bottom: 18px;
  box-shadow: 0 8px 24px rgba(47,89,255,0.12);
}
.header h1 { margin: 0; font-size: 22px; letter-spacing: -0.2px; font-weight: 700; }
.header p { margin: 6px 0 0 0; opacity: 0.95; color: rgba(255,255,255,0.92); font-size: 13px; }

/* Card surface (Material) */
.card {
  background: var(--card);
  border-radius: 12px;
  padding: 18px;
  margin-bottom: 18px;
  box-shadow: var(--surface-shadow);
  border: 1px solid rgba(15,23,42,0.04);
}

/* ... (other CSS rules for buttons, metrics, alerts) ... */
.stButton>button {
  background: linear-gradient(90deg, var(--primary), var(--accent));
  color: white;
  border: none;
  padding: 10px 14px;
  border-radius: 10px;
  font-weight: 600;
  box-shadow: 0 6px 16px rgba(47,89,255,0.16);
}

.alert {
  border-radius: 8px;
  padding: 12px;
  font-weight: 600;
  margin-bottom: 8px;
}
.alert.good { background: #e6f9f0; color: #065f46; border: 1px solid rgba(16,185,129,0.12); }
.alert.warn { background: #fffbeb; color: #92400e; border: 1px solid rgba(245,158,11,0.08); }
.alert.bad { background: #fff1f2; color: #9f1239; border: 1px solid rgba(239,68,68,0.08); }

/* Make Plotly backgrounds transparent within cards */
div.block-container .stPlotlyChart > div {
  background: transparent !important;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h1>AirAware — Air Quality Dashboard</h1><p>Milestone 4 — Material (corporate) UI • Light theme</p></div>', unsafe_allow_html=True)

# --------------------------
# Utilities (Cached)
# --------------------------

# Cache the model loading (so we only load from disk once)
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

# Cache the CSV loading
@st.cache_data
def load_csv_fast(path_or_buffer):
    # Try different encodings, as data files can be messy
    encs = ["utf-8", "windows-1252", "ISO-8859-1", "latin-1"]
    last_exc = None
    for enc in encs:
        try:
            return pd.read_csv(path_or_buffer, encoding=enc, low_memory=False)
        except Exception as e:
            last_exc = e
    # Final attempt
    try:
        return pd.read_csv(path_or_buffer, low_memory=False)
    except Exception:
        raise last_exc if last_exc is not None else Exception("Failed to read CSV")

# Cache filtering the dataframe by city
@st.cache_data
def get_city_df(df, location):
    s = df[df['location'] == location].copy()
    s = s.sort_values('date').reset_index(drop=True)
    return s

# Simple helper for MAE/RMSE
def mae_rmse(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# Helper to create lag features (used by some models)
def create_lag_features(series, lags=24):
    df = pd.DataFrame({'y': series.values})
    for i in range(1, lags+1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df = df.dropna().reset_index(drop=True)
    return df

# --------------------------
# Training / forecast functions
# --------------------------

# Each 'train_..._bundle' function does the same thing:
# 1. Checks if the library is installed.
# 2. Preps the data for that specific model.
# 3. Splits into train/validation sets.
# 4. Trains the model.
# 5. Calculates MAE/RMSE on the validation set.
# 6. Returns a 'bundle' (dictionary) with the model, metrics, and metadata.

def train_xgb_bundle(series_indexed, test_size):
    if not XGB_AVAILABLE:
        return None
    try:
        # Create time-based features (day of week, month, etc.)
        df = pd.DataFrame({'y': series_indexed.values})
        df['ds'] = series_indexed.index
        df['dayofweek'] = df['ds'].dt.dayofweek
        df['month'] = df['ds'].dt.month
        df['year'] = df['ds'].dt.year
        df['lag_7'] = df['y'].shift(7) # Add a 7-day lag feature
        df = df.dropna().reset_index(drop=True)
        if len(df) < 10:
            return None
        
        # Split data
        split = int(0.8*len(df))
        X_train = df[['dayofweek','month','year','lag_7']].iloc[:split]
        y_train = df['y'].iloc[:split]
        X_val = df[['dayofweek','month','year','lag_7']].iloc[split:]
        y_val = df['y'].iloc[split:]
        
        # Train model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_val)
        mae, rmse = mae_rmse(y_val, preds)
        return {'model': model, 'mae': mae, 'rmse': rmse, 'meta': {'type':'xgb'}}
    except Exception:
        return None

def train_prophet_bundle(series_indexed, test_size):
    if not PROPHET_AVAILABLE:
        return None
    try:
        # Prophet needs columns 'ds' (date) and 'y' (value)
        df_prop = series_indexed.reset_index().rename(columns={'date':'ds', series_indexed.name:'y'})
        if len(df_prop) < 10:
            return None
        
        # Split data
        split = int(0.8 * len(df_prop))
        train_df = df_prop.iloc[:split]
        test_df = df_prop.iloc[split:]
        
        # Train model
        m = Prophet(daily_seasonality=True, weekly_seasonality=True)
        m.fit(train_df[['ds','y']])
        
        # Evaluate
        preds = m.predict(test_df[['ds']])['yhat'].values
        mae, rmse = mae_rmse(test_df['y'].values, preds)
        return {'model': m, 'mae': mae, 'rmse': rmse, 'meta': {'type':'prophet'}}
    except Exception:
        return None

def train_arima_bundle(series_vals, test_size):
    if not ARIMA_AVAILABLE:
        return None
    try:
        if len(series_vals) < 20:
            return None
        
        # Train on all data *except* the test set
        m = ARIMA(series_vals[:-test_size], order=(5,1,0)).fit() # Using a standard (p,d,q) order
        
        # Evaluate by forecasting the test set period
        preds = m.forecast(steps=test_size)
        mae, rmse = mae_rmse(series_vals[-test_size:], preds)
        return {'model': m, 'mae': mae, 'rmse': rmse, 'meta': {'type':'arima'}}
    except Exception:
        return None

def train_lstm_bundle(series_vals, test_size, epochs=10):
    if not TF_AVAILABLE:
        return None
    try:
        # LSTM needs scaled data (0 to 1)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(series_vals.reshape(-1,1))
        
        time_step = 60 # Look back 60 time steps
        if len(scaled) <= time_step + test_size + 1:
            return None # Not enough data
        
        # Split into train/test
        train_data = scaled[:-test_size]
        test_data = scaled[len(train_data)-time_step:] # Test data needs the look-back period
        
        # Create sliding window datasets
        X_train, y_train = [], []
        for i in range(time_step, len(train_data)):
            X_train.append(train_data[i-time_step:i,0]); y_train.append(train_data[i,0])
        X_test, y_test = [], []
        for i in range(time_step, len(test_data)):
            X_test.append(test_data[i-time_step:i,0]); y_test.append(test_data[i,0])
        
        if not X_train:
            return None # Not enough data
        
        # Reshape for LSTM: [samples, time_steps, features]
        X_train = np.array(X_train).reshape(-1, time_step, 1)
        X_test = np.array(X_test).reshape(-1, time_step, 1)
        y_train = np.array(y_train); y_test = np.array(y_test)
        
        # Build and train the model
        model = Sequential([Input(shape=(time_step,1)), LSTM(50, return_sequences=True), LSTM(50), Dense(1)])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=0)
        
        # Evaluate
        preds_scaled = model.predict(X_test).flatten()
        # Invert scaling to get real values
        preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
        
        mae, rmse = mae_rmse(y_test_inv, preds)
        # Save the scaler in the bundle, we need it for forecasting
        return {'model': model, 'mae': mae, 'rmse': rmse, 'meta': {'scaler': scaler, 'time_step': time_step, 'type':'lstm'}}
    except Exception:
        return None

# This function takes a saved model 'bundle' and generates a forecast
def forecast_from_bundle(bundle, recent_series, horizon):
    if bundle is None:
        return None
    model = bundle.get('model')
    meta = bundle.get('meta', {})
    
    # Prophet forecast
    try:
        if PROPHET_AVAILABLE and isinstance(model, Prophet):
            future = model.make_future_dataframe(periods=horizon, freq='H')
            preds = model.predict(future)['yhat'].values[-horizon:]
            return np.array(preds)
    except Exception:
        pass
    
    # ARIMA forecast
    try:
        if ARIMA_AVAILABLE and hasattr(model, 'forecast'):
            return np.array(model.forecast(steps=horizon))
    except Exception:
        pass
    
    # XGBoost iterative forecast (one step at a time)
    try:
        if XGB_AVAILABLE and hasattr(model, 'predict'):
            meta_lags = 7 # Needs to match the lag feature it was trained on
            cur = list(recent_series[-meta_lags:])
            preds = []
            for i in range(horizon):
                last_ts = recent_series.index[-1]
                next_ts = pd.to_datetime(last_ts) + pd.Timedelta(hours=i+1)
                # Build the feature row for the *next* time step
                feat = pd.DataFrame([{'dayofweek': next_ts.dayofweek, 'month': next_ts.month, 'year': next_ts.year,
                                  'lag_7': cur[-7]}])
                p = model.predict(feat)[0]
                preds.append(p); cur.append(p) # Add prediction to history, repeat
            return np.array(preds)
    except Exception:
        pass
    
    # LSTM iterative forecast
    try:
        if TF_AVAILABLE and hasattr(model, 'predict'):
            scaler = meta.get('scaler'); time_step = meta.get('time_step', 60)
            arr = np.array(recent_series.values).astype(float)
            cur = list(arr[-time_step:]) # Get the last 'time_step' values
            preds = []
            for i in range(horizon):
                # Format the input for LSTM
                x = np.array(cur[-time_step:]).reshape(1, time_step, 1)
                # Predict one step
                p_scaled = model.predict(x).flatten()[0]
                # Un-scale the prediction
                p = scaler.inverse_transform([[p_scaled]])[0,0]
                preds.append(p); cur.append(p) # Add prediction to history, repeat
            return np.array(preds)
    except Exception:
        pass
    
    # Fallback: just repeat the last known value (naive forecast)
    if len(recent_series) > 0:
        return np.array([recent_series.values[-1]] * horizon)
    return None

# --------------------------
# Load default data if present
# --------------------------
DEFAULT_PATH = "All_Dashboards/data.csv"
data = None
if os.path.exists(DEFAULT_PATH):
    try:
        data = load_csv_fast(DEFAULT_PATH)
        st.sidebar.success("Loaded data.csv from app folder.")
    except Exception as e:
        st.sidebar.error(f"Failed to load data.csv: {e}")

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.markdown("## ⚙️ Controls", unsafe_allow_html=True)

# Admin checkbox
admin_mode = st.sidebar.checkbox("Admin Mode", value=False)

# Data uploader (only shows in Admin mode)
st.sidebar.markdown("---")
uploaded = None
if admin_mode:
    uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])
    st.sidebar.markdown("Upload a cleaned CSV with a station/location column and datetime.")

if uploaded is not None:
    try:
        uploaded.seek(0)
        data = load_csv_fast(uploaded) # Overwrite default data with uploaded file
        st.sidebar.success("Uploaded CSV loaded.")
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

# Stop if we have no data at all
if data is None:
    st.error("No data available. Place data.csv in app folder or upload via Admin.")
    st.stop()

# --- Data Normalization ---
# Clean column names (lowercase, strip whitespace)
data.columns = [c.strip().lower() for c in data.columns]
normalized_cols = list(data.columns)

# Flexible location detection
LOCATION_CANDIDATES = ['location','city','station','site','monitoring_station','station_name','location_name']
found_location_col = None
for col_name in LOCATION_CANDIDATES:
    if col_name in normalized_cols:
        found_location_col = col_name
        st.sidebar.info(f"Found location data in your '{col_name}' column.")
        break

if found_location_col is None:
    st.error(f"Upload failed. CSV must contain a location/city column.")
    st.stop()
if found_location_col != 'location':
    data = data.rename(columns={found_location_col: 'location'}) # Rename to 'location'

# Flexible date detection
date_col = None
for c in ['date', 'datetime', 'timestamp', 'sampling_date']:
    if c in data.columns:
        date_col = c
        break
if date_col:
    data['date'] = pd.to_datetime(data[date_col], errors='coerce')
else:
    # If no date, create synthetic dates (placeholder)
    n = len(data)
    maxp = min(n, 1000); dates = pd.date_range(end=dt.datetime.now(), periods=maxp)
    data['date'] = dates.repeat(n // maxp + 1)[:n].values
    st.warning("No date-like column detected — synthetic dates created.")

# Detect all numeric columns (pollutants)
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
# Also check for common names that might be 'object' type but are numeric
common = [c for c in ['pm2_5','pm10','so2','no2','rspm','spm','o3','aqi'] if c in data.columns and c not in numeric_cols]
numeric_cols += common
numeric_cols = sorted(list(set(numeric_cols)))
if not numeric_cols:
    st.error("No numeric pollutant columns detected in CSV.")
    st.stop()

# Get list of unique, clean city/location names
cities = sorted([str(c).strip() for c in data['location'].dropna().unique() if str(c).strip() != ""])
if not cities:
    st.error("No locations available to select.")
    st.stop()

# --- Sidebar selectors ---
city = st.sidebar.selectbox("Monitoring Station", cities, index=0)
time_range = st.sidebar.selectbox("Time Range", ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Data"])
pollutants = st.sidebar.multiselect("Pollutants (select one or more)", numeric_cols, default=[numeric_cols[0]])
if not pollutants:
    st.sidebar.error("Please select at least one pollutant.")
    st.stop()
gauge_pollutant = st.sidebar.selectbox("AQI gauge pollutant (choose one)", pollutants)
horizon_opt = st.sidebar.selectbox("Forecast Horizon", ["24 Hours", "48 Hours", "72 Hours", "168 Hours (7 days)"])
horizon_hours = int(horizon_opt.split()[0]) # Convert "24 Hours" to 24

if st.sidebar.button("Update Dashboard"):
    st.experimental_rerun()

# --- Admin training controls (if admin_mode) ---
train_button = None
if admin_mode:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Admin — Train models")
    train_button = st.sidebar.button("Train best models for selected city & pollutants")
    overwrite_models = st.sidebar.checkbox("Overwrite existing saved models", value=False)
    lstm_epochs = st.sidebar.number_input("LSTM epochs", min_value=1, max_value=200, value=10)

# --------------------------
# Training workflow (if button pressed)
# --------------------------
if admin_mode and train_button:
    st.sidebar.info("Training best models for selected city & pollutants...")
    city_df = get_city_df(data, city)
    if city_df.empty:
        st.sidebar.error("No records for selected city.")
    else:
        # Show progress bar and log area
        prog = st.sidebar.progress(0)
        log = st.sidebar.empty()
        total_tasks = len(pollutants) * 4 # 4 models per pollutant
        step = 0
        created_files = []
        
        # Loop over each pollutant selected
        for pol in pollutants:
            log.info(f"Preparing data for pollutant: {pol}")
            if pol not in city_df.columns or city_df[pol].dropna().empty:
                log.warning(f"No numeric data for {pol} — skipping.")
                step += 4; prog.progress(min(100, int(step/total_tasks*100)))
                continue
            
            # Get the time series data for this pollutant
            ts = city_df.set_index('date')[pol].dropna()
            n = len(ts)
            if n < 30: # Need at least 30 data points
                log.warning(f"Not enough data to train for {pol} (need >=30 rows).")
                step += 4; prog.progress(min(100, int(step/total_tasks*100)))
                continue
            
            TEST_SIZE = max(1, int(0.2 * n)) # 20% validation set
            trained_models = {}
            
            # --- Try training each model ---
            # XGBoost
            step += 1; prog.progress(min(100, int(step/total_tasks*100)))
            log.info(f"[{pol}] Training XGBoost (if available)...")
            out = train_xgb_bundle(ts, TEST_SIZE)
            if out: trained_models['XGBoost'] = out; log.success(f"[{pol}] XGBoost done.")

            # Prophet
            step += 1; prog.progress(min(100, int(step/total_tasks*100)))
            log.info(f"[{pol}] Training Prophet (if available)...")
            out = train_prophet_bundle(ts.rename(pol), TEST_SIZE)
            if out: trained_models['Prophet'] = out; log.success(f"[{pol}] Prophet done.")
            
            # ARIMA
            step += 1; prog.progress(min(100, int(step/total_tasks*100)))
            log.info(f"[{pol}] Training ARIMA (if available)...")
            out = train_arima_bundle(ts.values, TEST_SIZE)
            if out: trained_models['ARIMA'] = out; log.success(f"[{pol}] ARIMA done.")

            # LSTM
            step += 1; prog.progress(min(100, int(step/total_tasks*100)))
            log.info(f"[{pol}] Training LSTM (if available)...")
            out = train_lstm_bundle(ts.values, TEST_SIZE, epochs=int(lstm_epochs))
            if out: trained_models['LSTM'] = out; log.success(f"[{pol}] LSTM done.")

            # --- Find the best model (lowest MAE) ---
            ranking = {}
            for nm, b in trained_models.items():
                if b and 'mae' in b:
                    ranking[nm] = b['mae']
            
            if not ranking:
                log.warning(f"[{pol}] No successful models trained; skipping save.")
                continue
            
            best_name = min(ranking, key=ranking.get)
            best_bundle = trained_models[best_name]
            
            # --- Save the best model to disk ---
            save_fname = os.path.join(MODELS_DIR, f"{city}__{pol}__best.joblib")
            if os.path.exists(save_fname) and not overwrite_models:
                log.warning(f"[{pol}] best model exists: {save_fname} (enable overwrite to replace).")
            else:
                joblib.dump({'model_name': best_name, 'bundle': best_bundle}, save_fname)
                created_files.append(save_fname)
                log.success(f"[{pol}] Saved best model: {best_name} -> {save_fname}")
        
        prog.progress(100); time.sleep(0.5)
        st.sidebar.success("Training process finished.")
        
        # Zip the created models for easy download
        if created_files:
            zipname = os.path.join(MODELS_DIR, f"{city}__{'_'.join(pollutants)}__models.zip")
            with zipfile.ZipFile(zipname, 'w') as zf:
                for f in created_files:
                    zf.write(f, arcname=os.path.basename(f))
            st.sidebar.success(f"Saved trained artifacts: {zipname}")

# --------------------------
# Load saved models for selected pollutants
# --------------------------
# This runs every time (not just in admin) to load models for forecasting
saved_models = {}
for pol in pollutants:
    fname = os.path.join(MODELS_DIR, f"{city}__{pol}__best.joblib")
    if os.path.exists(fname):
        saved_models[pol] = load_model(fname) # Uses the cached loader
    else:
        saved_models[pol] = None

# --------------------------
# Data prep for dashboard
# --------------------------
city_df = get_city_df(data, city) # Get the filtered data for the selected city

# compute current AQI-like metrics (using the gauge_pollutant)
try:
    latest = float(city_df[gauge_pollutant].dropna().iloc[-1])
except Exception:
    latest = 0.0 # Default to 0 if no data

# Dominant pollutant (max recent mean)
try:
    recent = city_df.tail(72) # Look at last 3 days
    dom = recent[pollutants].mean().idxmax()
except Exception:
    dom = gauge_pollutant

# Trend: compare last window to prior
try:
    avg_now = recent[gauge_pollutant].mean()
    avg_prev = city_df.tail(168).head(72)[gauge_pollutant].mean() if len(city_df) > 168 else avg_now
    trend = "↗️ Increasing" if avg_now > avg_prev else "↘️ Decreasing"
except Exception:
    trend = "—"

# --------------------------
# Layout: Main Dashboard
# --------------------------
tab1_dash, tab2_details, tab3_admin = st.tabs(["Dashboard", "Pollutant Details", "Model & Admin Status"])

# TAB 1: Dashboard
with tab1_dash:
    preds_store = {} # To store forecast arrays

    # --- Alerts + Overview row ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Alert Notifications", anchor=None)
    
    # --- Forecast logic ---
    # We need to generate forecasts *before* we can show alerts
    fig_all = go.Figure() # This figure will hold all history + forecasts
    
    for pol in pollutants:
        series = city_df.set_index('date')[pol].dropna() if pol in city_df.columns else pd.Series(dtype=float)
        if series.empty:
            continue
        
        # Filter historical data by time range
        if time_range == "Last 24 Hours": cutoff = pd.Timestamp.now() - pd.Timedelta(hours=24)
        elif time_range == "Last 7 Days": cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
        elif time_range == "Last 30 Days": cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
        else: cutoff = None
        plot_series = series[series.index >= cutoff] if cutoff is not None else series
        
        # Add historical data to the chart
        if not plot_series.empty:
            fig_all.add_trace(go.Scatter(x=plot_series.index, y=plot_series.values, mode='lines+markers', name=f"{pol} (hist)"))
        
        # Get the saved model for this pollutant
        bundle = saved_models.get(pol)
        preds = None
        if bundle is not None:
            # We have a trained model
            try:
                b = bundle.get('bundle')
                recent_series = series[-200:] # Get recent data for the model
                preds = forecast_from_bundle(b, recent_series, horizon_hours) # Generate forecast
                
                if preds is not None:
                    last_ts = recent_series.index[-1]
                    pred_index = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq='H')
                    # Add forecast data to the chart
                    fig_all.add_trace(go.Scatter(x=pred_index, y=preds, mode='lines+markers', name=f"{pol} (forecast)"))
                    preds_store[pol] = (pred_index, preds) # Save preds
            except Exception:
                preds = None
        else:
            # No trained model, use naive forecast (repeat last value)
            if not series.empty:
                last = series.values[-1]
                preds = np.array([last] * horizon_hours)
                last_ts = series.index[-1]
                pred_index = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq='H')
                fig_all.add_trace(go.Scatter(x=pred_index, y=preds, mode='lines+markers', name=f"{pol} (naive)"))
                preds_store[pol] = (pred_index, preds)

    # --- Render alert based on predicted peaks ---
    all_peaks = []
    for pol, val in preds_store.items():
        if val is not None:
            peaks = np.nanmax(val[1]) # Find max value in forecast
            all_peaks.append(peaks)
            
    if all_peaks:
        peak = max(all_peaks) # Find the highest peak of *all* pollutants
        # Show an alert based on the peak value
        if peak > 350:
            st.markdown('<div class="alert bad">🔥 Severe pollution predicted — take urgent action</div>', unsafe_allow_html=True)
        elif peak > 200:
            st.markdown('<div class="alert warn">⚠️ High pollution expected — be cautious</div>', unsafe_allow_html=True)
        elif peak > 100:
            st.markdown('<div class="alert warn">⚠️ Moderate pollution expected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert good">✅ Air quality looks good</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert warn">No forecasts available — train models in Admin mode</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) # End card

    # --- Overview Metrics ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Overview")
    col1, col2, col3 = st.columns([1,1,1])
    col1.metric("Current Value", f"{latest:.1f}")
    col2.metric("Dominant Pollutant", dom)
    col3.metric("Trend", trend)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Main Dashboard Grid ---
    top_left, top_right = st.columns([0.6, 1.4], gap="large")
    
    with top_left:
        # --- AQI Gauge ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Current Air Quality")
        max_val = max(100, latest * 2) # Dynamic max for the gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest,
            number={'suffix': "", 'font': {'size': 28}},
            gauge={
                'axis': {'range': [0, max_val]},
                'bar': {'color': '#2f59ff'}, # Blue bar
                # Color steps
                'steps': [
                    {'range': [0, 50], 'color': '#22c55e'}, # Green
                    {'range': [50, 100], 'color': '#f59e0b'}, # Yellow
                    {'range': [100, 200], 'color': '#f97316'}, # Orange
                    {'range': [200, max_val], 'color': '#ef4444'}, # Red
                ]
            }
        ))
        fig_g.update_layout(height=300, margin=dict(l=6,r=6,t=6,b=6), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with top_right:
        # --- Forecast Chart ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Forecasts (historical + predicted)")
        if len(fig_all.data) == 0:
            st.write("No historical/forecast data available.")
        else:
            fig_all.update_layout(height=350, margin=dict(l=6,r=6,t=6,b=6), hovermode='x unified', legend=dict(orientation="h", y=1.08), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_all, use_container_width=True)
            
            # Add download button for the forecast data
            if preds_store:
                export_rows = []
                for pol, (idx, preds) in preds_store.items():
                    for t, v in zip(idx, preds):
                        export_rows.append({'city': city, 'pollutant': pol, 'datetime': t, 'predicted': float(v)})
                df_export = pd.DataFrame(export_rows)
                csv = df_export.to_csv(index=False).encode('utf-8')
                st.download_button("Download Forecast CSV", csv, file_name=f"{city}_forecast.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

    # Lower row: trends + alerts list
    bot_left, bot_right = st.columns([1.6,1], gap="large")
    with bot_left:
        # --- Historical Trends ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Pollutant Trends")
        if city_df.empty:
            st.write("No data for this city.")
        else:
            # This is the same plot as in the forecast, but *just* historical
            series_plot = city_df.set_index('date')
            if cutoff: series_plot = series_plot[series_plot.index >= cutoff]
            
            fig = go.Figure()
            plotted = False
            for pol in pollutants:
                if pol in series_plot.columns and not series_plot[pol].dropna().empty:
                    fig.add_trace(go.Scatter(x=series_plot.index, y=series_plot[pol].values, mode='lines+markers', name=pol))
                    plotted = True
            if plotted:
                fig.update_layout(height=320, margin=dict(l=6,r=6,t=6,b=6), hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No numeric values for selected pollutants in this timeframe.")
        st.markdown('</div>', unsafe_allow_html=True)

    with bot_right:
        # --- Alert List ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Alert Notifications")
        # Display the alert messages based on peaks
        if preds_store:
            for pol, (idx, preds) in preds_store.items():
                peak_val = float(np.nanmax(preds))
                when = str(idx[np.nanargmax(preds)])
                if peak_val > 200:
                    st.markdown(f"<div class='alert bad'>⚠️ {pol}: High peak ~{peak_val:.0f} expected</div>", unsafe_allow_html=True)
                elif peak_val > 100:
                    st.markdown(f"<div class='alert warn'>⚠ {pol}: Moderate peak ~{peak_val:.0f}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='alert good'>✅ {pol}: No major peaks predicted</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert warn'>No forecasts available</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Pollutant Details
with tab2_details:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Pollutant Details & Analysis")
    
    # --- Main Plot (same as on dashboard) ---
    if city_df.empty:
        st.write("No data for this city.")
    else:
        series_plot = city_df.set_index('date')
        if cutoff: series_plot = series_plot[series_plot.index >= cutoff]
        fig = go.Figure()
        plotted = False
        for pol in pollutants:
            if pol in series_plot.columns and not series_plot[pol].dropna().empty:
                fig.add_trace(go.Scatter(x=series_plot.index, y=series_plot[pol].values, mode='lines+markers', name=pol))
                plotted = True
        if plotted:
            fig.update_layout(height=420, margin=dict(l=6,r=6,t=6,b=6), hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No numeric values in this timeframe.")
    st.markdown('</div>', unsafe_allow_html=True)

    det1, det2 = st.columns(2, gap="large")
    with det1:
        # --- Correlation Heatmap ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.expander("Correlation between selected pollutants", expanded=True):
            try:
                corr_df = city_df[pollutants].corr()
                fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale=px.colors.sequential.Blues)
                fig_corr.update_layout(height=300, margin=dict(l=6,r=6,t=40,b=6), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.write("Correlation plot error:", e)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with det2:
        # --- Hourly Average (Diurnal) Plot ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.expander("Daily Pattern (average by hour)", expanded=True):
            try:
                city_df['hour'] = city_df['date'].dt.hour
                hourly = city_df.groupby('hour')[pollutants].mean().reset_index()
                fig_h = px.line(hourly, x='hour', y=pollutants, markers=True)
                fig_h.update_layout(height=300, margin=dict(l=6,r=6,t=40,b=6), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_h, use_container_width=True)
            except Exception as e:
                st.write("Daily pattern error:", e)
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: Model & Admin Status
with tab3_admin:
    col_bottom_left, col_bottom_right = st.columns(2, gap="large")
    with col_bottom_left:
        # --- Model Status Card ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model Status")
        # Check which models are loaded
        for pol in pollutants:
            sm = saved_models.get(pol)
            if sm is None:
                st.write(f"- **{pol}**: No saved best model")
            else:
                model_name = sm.get('model_name', 'unknown')
                st.write(f"- **{pol}**: saved best = **{model_name}**")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_bottom_right:
        # --- Admin Info Card ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Admin / Training")
        st.write("Use Admin toggle in the sidebar to upload dataset and train models.")
        st.write("Saved model files (recent):")
        # List files in the /models directory
        model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib") or f.endswith(".pkl")], reverse=True)
        if model_files:
            for f in model_files:
                try:
                    mtime = dt.datetime.fromtimestamp(os.path.getmtime(os.path.join(MODELS_DIR, f)))
                    st.write(f"- `{f}` (saved: {mtime.strftime('%Y-%m-%d %H:%M')})")
                except Exception:
                    st.write(f"- `{f}`")
        else:
            st.write("No model artifacts saved yet.")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Built with Streamlit • Models in /models • Admin mode trains XGBoost, Prophet, ARIMA, LSTM (if installed)</div>', unsafe_allow_html=True)
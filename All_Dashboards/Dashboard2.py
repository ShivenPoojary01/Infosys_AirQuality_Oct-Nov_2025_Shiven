import os
import json
import hashlib  # Used for hashing dataframes for caching
import io

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt  # For data viz
import matplotlib.pyplot as plt # For Prophet components plot

# Models
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Keras / Tensorflow for LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(layout="wide", page_title="Air Quality Forecast Engine", page_icon="🌫️")

# ---------------------------
# Altair Theme (Forced Default/Light)
# ---------------------------
alt.themes.enable("default")

# ---------------------------
# CSS / Fonts (Forced Light Mode)
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

/* Set global font */
html, body, [data-testid="stApp"], h1, h2, h3, h4, h5, h6, p, li, 
[data-testid="stMarkdown"], [data-testid="stMetric"], [data-testid="stRadio"],
[data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stButton"] {
    font-family: 'Poppins', sans-serif;
}

/* App Background */
.stApp {
    background-color: #f8faff; /* Lighter, bluer background */
    color: #0f172a;
}

/* --- Gradient Header --- */
.header-bar {
    background: linear-gradient(90deg, #0052D4, #4364F7); /* Blue gradient */
    padding: 14px 20px;
    border-radius: 8px;
    margin-bottom: 20px;
}
.header-bar h1 { color: #ffffff; font-weight: 700; margin: 0; font-size: 28px; }
.header-bar p { color: #e0eaff; margin: 0; font-size: 16px; }

/* --- Enhanced Card Styling (Base) --- */
div[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 24px;
    min-height: 450px; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
}

/* --- Card Hover Effect --- */
div[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] > [data-testid="stContainer"]:hover {
    transform: translateY(-5px); /* Lift the card */
    box-shadow: 0 8px 20px rgba(0, 82, 212, 0.12); /* Softer, blue-tinted shadow */
    border-color: #4364F7; /* Accent border color */
}

/* --- Card Title Styling (Accent Color) --- */
div[data-testid="stContainer"] h3 {
    font-size: 20px;
    font-weight: 600;
    color: #0052D4; /* Dark blue accent */
    margin-top: 0;
    margin-bottom: 16px;
}

/* --- Sidebar & Table Styling --- */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    padding: 16px;
    border-right: 1px solid #e6e9ef;
}
[data-testid="stDataFrame"] thead th { 
    background-color: #f3f4f6; 
    color: #0f172a; 
}

/* --- Styled Download Button --- */
[data-testid="stDownloadButton"] button {
    background-color: #0052D4; /* Match accent */
    color: #ffffff;
    border: 1px solid #0052D4;
}
[data-testid="stDownloadButton"] button:hover {
    background-color: #0041a8; /* Darker on hover */
    border-color: #0041a8;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------
# Utility helpers
# ---------------------------
def df_hash(df: pd.DataFrame) -> str:
    """Creates a unique hash from a DataFrame for caching."""
    j = df.to_json(date_format='iso', orient='split')
    return hashlib.md5(j.encode('utf-8')).hexdigest()

class StreamlitCallback(Callback):
    """Custom Keras callback to update a Streamlit progress bar."""
    def __init__(self, progress_bar, epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.epoch_count = epochs

    # On epoch end, update the progress bar
    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epoch_count
        try:
            # Update the bar
            self.progress_bar.progress(progress, text=f"LSTM Epoch {epoch+1}/{self.epoch_count}")
        except Exception:
            # Failsafe if bar doesn't exist anymore
            pass

def create_dataset(dataset, look_back=1):
    """Converts time series into a sliding window dataset for LSTM."""
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# --- Helper functions for Forecast Accuracy ---
def lstm_iterative_predict(model, history_series, look_back, steps):
    """
    Iteratively predicts 'steps' ahead.
    We predict one step, add it to the history, then predict the next step.
    This simulates a real-world forecast.
    """
    if model is None:
        return None
        
    # Fit scaler to history for consistent scaling
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(history_series.reshape(-1,1))
    inputs = scaler.transform(history_series.reshape(-1,1))
    
    preds = []
    # Get the last 'look_back' sequence from history
    seq = inputs[-look_back:].reshape(1, look_back, 1)
    
    for _ in range(steps):
        # Predict 1 step
        p = model.predict(seq, verbose=0)
        preds.append(p.flatten()[0])
        # Append the new prediction and drop the oldest value
        seq = np.append(seq[:,1:,:], p.reshape(1,1,1), axis=1)
        
    preds = np.array(preds).reshape(-1,1)
    # Invert the scaling to get real values
    preds_inv = scaler.inverse_transform(preds).flatten()
    return preds_inv

# ---------------------------
# Forecast Accuracy Logic
# ---------------------------
def compute_horizon_accuracy(train_df, full_df, test_df, models_dict, horizons, lstm_lookback=30):
    """Calculates RMSE for each model at different future horizons (e.g., 1, 7, 30 days)."""
    
    all_model_names = ["Prophet", "ARIMA", "LSTM"]
    results = {name: [] for name in all_model_names}
    
    full_series = full_df['y'].values
    
    # Loop through each horizon (e.g., 1 day, 3 days, 7 days...)
    for h in horizons:
        # Prophet
        try:
            if 'Prophet' in models_dict and models_dict['Prophet'] is not None:
                m = models_dict['Prophet']
                future_dates = test_df['ds'].iloc[:h].tolist() # Get first 'h' test dates
                fc = m.predict(pd.DataFrame({'ds': future_dates}))
                preds = fc['yhat'].values
                actuals = test_df['y'].iloc[:len(preds)].values
                rmse = np.sqrt(mean_squared_error(actuals, preds)) if len(preds)>0 else np.nan
                results['Prophet'].append(rmse)
            else:
                results['Prophet'].append(np.nan) # Append NaN if model wasn't run
        except Exception:
            results['Prophet'].append(np.nan)

        # ARIMA
        try:
            if 'ARIMA' in models_dict and models_dict['ARIMA'] is not None:
                ar = models_dict['ARIMA']
                preds = ar.forecast(steps=h) # Forecast 'h' steps ahead
                actuals = test_df['y'].iloc[:len(preds)].values
                rmse = np.sqrt(mean_squared_error(actuals, preds)) if len(preds)>0 else np.nan
                results['ARIMA'].append(rmse)
            else:
                results['ARIMA'].append(np.nan)
        except Exception:
            results['ARIMA'].append(np.nan)

        # LSTM
        try:
            if 'LSTM' in models_dict and models_dict['LSTM'] is not None:
                lstm_model = models_dict['LSTM']
                if len(full_series) >= lstm_lookback + 1:
                    # Use the iterative predictor
                    preds = lstm_iterative_predict(lstm_model, full_series, lstm_lookback, h)
                    actuals = test_df['y'].iloc[:len(preds)].values
                    rmse = np.sqrt(mean_squared_error(actuals, preds)) if len(preds)>0 else np.nan
                    results['LSTM'].append(rmse)
                else:
                    results['LSTM'].append(np.nan)
            else:
                results['LSTM'].append(np.nan)
        except Exception:
            results['LSTM'].append(np.nan)
            
    return results

# ---------------------------
# Data loading & preprocessing
# ---------------------------
@st.cache_data
def load_raw_data(path="human_readable_air_quality_CITYWISE.csv"):
    """Loads the raw CSV data."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
    return df

@st.cache_data
def get_city_list(_df):
    """Gets a unique, sorted list of cities from the dataframe."""
    if _df.empty:
        return []
    return sorted(_df['location'].unique())

@st.cache_data
def preprocess_data_cached(df_json: str, city: str, pollutant_col: str):
    """
    Takes raw data, filters, cleans, interpolates, and splits it.
    This is the main data prep pipeline.
    """
    df = pd.read_json(df_json, convert_dates=['date'], orient='split')
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, 0, 0
    
    # Filter by city
    city_df = df[df['location'] == city].copy()
    if city_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, 0, 0
        
    city_df.set_index('date', inplace=True)
    if pollutant_col not in city_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, 0, 0
    
    # Resample to Daily ('D') mean. This is a key step.
    data_series = city_df[pollutant_col].resample('D').mean()
    
    # Fill missing values (e.g., weekends)
    data_series = data_series.interpolate(method='time').ffill().bfill()
    
    # Format for Prophet/ARIMA (Prophet needs 'ds' and 'y')
    clean_data = data_series.to_frame(name='y')
    clean_data['ds'] = clean_data.index
    clean_data.reset_index(drop=True, inplace=True)
    
    # Split data: 80% train, 20% test
    total_points = len(clean_data)
    split_index = int(total_points * 0.8)
    
    if split_index < 2: # Not enough data to train
        return pd.DataFrame(), pd.DataFrame(), clean_data, total_points, 0, max(0, total_points - split_index)
        
    train_data = clean_data.iloc[:split_index]
    test_data = clean_data.iloc[split_index:]
    
    return train_data, test_data, clean_data, total_points, len(train_data), len(test_data)

# ---------------------------
# Model training
# ---------------------------

@st.cache_resource
def _train_prophet_cache(train_json: str, seed: int = 0):
    """Internal cached function for Prophet."""
    df = pd.read_json(train_json, orient='split', convert_dates=['ds'])
    model = Prophet()
    model.fit(df)
    return model

def train_prophet(train_df: pd.DataFrame):
    """Wrapper function to train Prophet model, using cache."""
    if train_df.empty:
        return None
    train_json = train_df.to_json(date_format='iso', orient='split')
    # The cache checks `train_json`. If it's seen it before, it returns
    # the cached model instead of re-training.
    return _train_prophet_cache(train_json)

@st.cache_resource
def _train_arima_cache(y_json: str):
    """Internal cached function for ARIMA."""
    y = json.loads(y_json)
    ser = pd.Series(y)
    model = ARIMA(ser, order=(5, 1, 0)) # Using a standard p,d,q order
    fitted = model.fit()
    return fitted

def train_arima(train_df: pd.DataFrame):
    """Wrapper function to train ARIMA model, using cache."""
    if train_df.empty:
        return None
    y_json = json.dumps(train_df['y'].tolist())
    return _train_arima_cache(y_json)

@st.cache_resource
def _train_lstm_cache(scaled_json: str, look_back: int, epochs: int, seed: int = 0):
    """Internal cached function for LSTM."""
    X = np.array(json.loads(scaled_json))
    X = np.array(X).reshape(-1, 1)
    
    # Create the sliding window
    X_train, y_train = create_dataset(X, look_back)
    if X_train.shape[0] == 0: # Not enough data
        return None, None
        
    # Reshape for LSTM [samples, time_steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
    return model, None

def train_lstm(train_df: pd.DataFrame, look_back=30, epochs=20, quick_mode=False):
    """Wrapper function to train LSTM model, using cache."""
    if train_df.empty:
        return None, None
        
    # LSTM requires data to be scaled (usually 0 to 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(train_df['y'].values.reshape(-1, 1)).flatten().tolist()
    scaled_json = json.dumps(scaled)
    
    e = max(1, int(epochs))
    lb = max(1, int(look_back))
    
    model, _ = _train_lstm_cache(scaled_json, lb, e)
    return model, scaler

# ---------------------------
# Main app setup
# ---------------------------
raw_df = load_raw_data()
if raw_df.empty:
    st.title("Air Quality Forecast Engine")
    st.error("`human_readable_air_quality_CITYWISE.csv` not found in the app folder. Please upload your cleaned CSV.")
    st.stop()

cities = get_city_list(raw_df)
# Find columns that we can actually forecast
pollutant_options = [c for c in ["avg_rspm", "avg_so2", "avg_no2", "pm2.5", "pm10"] if c in raw_df.columns]
if not pollutant_options:
    st.error("No compatible pollutant columns found in CSV. Expected: avg_rspm, avg_so2, avg_no2, pm2.5, or pm10.")
    st.stop()

model_options = ["Prophet", "ARIMA", "LSTM"]

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Forecast Controls")
    # (Theme Toggle removed - forcing Light Mode)

    # Set Hyderabad as default if it exists
    default_index = 0
    if "Hyderabad" in cities:
        default_index = cities.index("Hyderabad")
    selected_city = st.selectbox("Select City", options=cities, index=default_index)

    # Select pollutants and models to run
    selected_pollutants = st.multiselect("Select Pollutant(s)", options=pollutant_options, default=pollutant_options[:1])
    selected_models = st.multiselect("Select Model(s)", options=model_options, default=model_options)

    st.markdown("---")
    # LSTM-specific controls
    quick_mode = st.checkbox("⚡ Quick Mode (faster)", value=False)
    lstm_epochs = st.slider("LSTM Epochs", 1, 100, 20 if not quick_mode else 5)
    lstm_lookback = st.number_input("LSTM Look-back (days)", min_value=1, max_value=60, value=30 if not quick_mode else 15)
    st.markdown("---")

# Initialize session state to store results
if 'perf_df' not in st.session_state:
    st.session_state.perf_df = pd.DataFrame()
if 'all_plot_data' not in st.session_state:
    st.session_state.all_plot_data = {}
if 'data_info' not in st.session_state:
    st.session_state.data_info = {}
if 'selected_pollutants_cache' not in st.session_state:
    st.session_state.selected_pollutants_cache = []
if 'run_success' not in st.session_state:
    st.session_state.run_success = False

# ---------------------------
# Header Bar
# ---------------------------
st.markdown(f"""
<div class="header-bar">
    <h1>Air Quality Forecast Engine</h1>
    <p>Milestone 2: Working Application (Weeks 3–4)</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------
# Auto-run logic (The main "engine")
# ---------------------------
# This runs if the user has selected at least one pollutant and model
if selected_pollutants and selected_models:
    with st.spinner(f"Running analysis for {selected_city}... This may take a moment."):
        all_model_results = []
        all_plot_data = {}
        data_info = {}
        # Store the selected pollutants in cache so the UI doesn't break if user deselects them
        st.session_state.selected_pollutants_cache = selected_pollutants
        df_json = raw_df.to_json(date_format='iso', orient='split')

        # --- Outer loop: iterate over each selected pollutant ---
        for pollutant in selected_pollutants:
            # 1. Get and prep data
            train_data, test_data, full_clean_data, total_pts, train_pts, test_pts = preprocess_data_cached(df_json, selected_city, pollutant)
            data_info[pollutant] = {"Total": total_pts, "Train": train_pts, "Test": test_pts}
            
            if train_data.empty or test_data.empty or total_pts < 5:
                st.warning(f"Not enough data for {pollutant}. Skipping.")
                continue
                
            actuals = test_data['y'].values
            
            # Prep data for plotting
            actuals_df = full_clean_data.rename(columns={'y': 'Actuals'})
            actuals_df['Model'] = 'Actuals'
            all_plot_data[pollutant] = [actuals_df] # This will hold 'Actuals' + all model preds
            trained_models = {} # To store trained models for the accuracy chart

            # --- Inner loop: iterate over each selected model ---
            for model_name in selected_models:
                try:
                    preds = None
                    # 2. Train model
                    if model_name == "Prophet":
                        model = train_prophet(train_data[['ds', 'y']])
                        trained_models['Prophet'] = model
                        
                        # 3. Predict on test set
                        test_future_df = pd.DataFrame({'ds': test_data['ds']})
                        forecast = model.predict(test_future_df)
                        preds = forecast['yhat'].values
                        
                    elif model_name == "ARIMA":
                        model = train_arima(train_data[['y']])
                        trained_models['ARIMA'] = model
                        
                        # 3. Predict
                        preds = model.forecast(steps=len(test_data))
                        
                    elif model_name == "LSTM":
                        model, scaler = train_lstm(train_data[['y']], look_back=lstm_lookback, epochs=lstm_epochs, quick_mode=quick_mode)
                        trained_models['LSTM'] = model
                        
                        # 3. Predict (this is more complex for LSTM)
                        if model:
                            # We need to feed the model the last 'look_back' days from *before* the test set
                            inputs = full_clean_data['y'].values
                            start_idx = len(inputs) - len(test_data) - lstm_lookback
                            if start_idx < 0:
                                st.warning(f"LSTM for {pollutant} skipped: not enough historical points for look_back.")
                                continue
                            
                            inputs_seq = inputs[start_idx:].reshape(-1, 1)
                            
                            # We must scale the test data using the *full* dataset's range
                            scaler_for_inv = MinMaxScaler(feature_range=(0, 1))
                            scaler_for_inv.fit(full_clean_data['y'].values.reshape(-1,1))
                            inputs_scaled = scaler_for_inv.transform(inputs_seq)
                            
                            # Create sliding window for test set
                            X_test = []
                            for i in range(lstm_lookback, len(inputs_scaled)):
                                X_test.append(inputs_scaled[i-lstm_lookback:i, 0])
                            X_test = np.array(X_test)
                            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                            
                            # Predict
                            test_preds_scaled = model.predict(X_test, verbose=0)
                            # Invert scaling to get real values
                            preds = scaler_for_inv.inverse_transform(test_preds_scaled).flatten()
                        else:
                            st.warning(f"LSTM for {pollutant} skipped: not enough data to form sequences.")
                            continue

                    # 4. Calculate metrics and store results
                    if preds is not None:
                        pred_len = len(preds)
                        actuals_sliced = actuals[:pred_len]
                        
                        rmse = np.sqrt(mean_squared_error(actuals_sliced, preds))
                        mae = mean_absolute_error(actuals_sliced, preds)
                        
                        # Store metrics
                        all_model_results.append({
                            'Pollutant': pollutant.upper(), 'Model': model_name,
                            'RMSE': float(rmse), 'MAE': float(mae)
                        })
                        
                        # Store plot data
                        preds_df = pd.DataFrame({
                            'ds': test_data['ds'].values[:pred_len],
                            'Actuals': preds, 'Model': model_name
                        })
                        all_plot_data[pollutant].append(preds_df)
                        
                except Exception as e:
                    st.error(f"Training for {model_name} on {pollutant} failed: {e}")
            
            # Store the trained models for this pollutant
            all_plot_data.setdefault('_models', {})[pollutant] = trained_models
        
        # --- After all loops, save results to session state ---
        if not all_model_results:
            # If nothing worked, clear the state
            st.session_state.perf_df = pd.DataFrame()
            st.session_state.all_plot_data = {}
            st.session_state.data_info = {}
            st.session_state.run_success = False
        else:
            # Save all results
            st.session_state.perf_df = pd.DataFrame(all_model_results)
            st.session_state.all_plot_data = all_plot_data
            st.session_state.data_info = data_info
            st.session_state.run_success = True # Flag to show success message

elif not selected_pollutants or not selected_models:
    # If user deselects all models/pollutants, clear the results
    st.session_state.perf_df = pd.DataFrame()
    st.session_state.all_plot_data = {}
    st.session_state.data_info = {}
    st.session_state.run_success = False

# ---------------------------
# Results display (2x2 GRID LAYOUT)
# ---------------------------

# Load results from session state
perf_df = st.session_state.perf_df
all_plot_data = st.session_state.all_plot_data
data_info = st.session_state.data_info
selected_pollutants_cache = st.session_state.selected_pollutants_cache

# Show a one-time success message
if st.session_state.run_success:
    st.success("Analysis complete!")
    st.session_state.run_success = False # Reset flag

# Define the 2x2 grid
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# --- TOP-LEFT: Model Performance (Bar Chart) ---
with row1_col1:
    with st.container(border=True): # This is one of the styled "cards"
        c1_col1, c1_col2 = st.columns([2, 1])
        with c1_col1:
            st.markdown("<h3>Model Performance</h3>", unsafe_allow_html=True)
        
        if perf_df.empty:
            st.info("Select pollutant(s) and model(s) to run analysis.")
        else:
            with c1_col2:
                # Radio button to switch between metrics
                metric_choice = st.radio("Metric", ["RMSE", "MAE"], horizontal=True, key='metric_radio_1')
            
            # Prepare data for Altair chart
            chart_df = perf_df.melt(id_vars=['Pollutant', 'Model'], value_vars=['RMSE', 'MAE'], var_name='Metric', value_name='Value')
            chart_df = chart_df[chart_df['Metric'] == metric_choice]
            
            if chart_df.empty:
                st.info("No performance data to show.")
            else:
                # Create the Altair bar chart
                chart = alt.Chart(chart_df).mark_bar(cornerRadius=6).encode(
                    x=alt.X('Model:N', title=None, axis=None),
                    y=alt.Y('Value:Q', title=metric_choice),
                    color=alt.Color('Model:N', legend=alt.Legend(title="Model", orient="bottom")),
                    # Create a separate column for each pollutant
                    facet=alt.Facet('Pollutant:N', columns=3, header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
                    tooltip=[alt.Tooltip('Pollutant:N'), alt.Tooltip('Model:N'), alt.Tooltip('Value:Q', format=".2f")]
                ).properties(height=300).configure_view(
                    strokeOpacity=0 # Remove border
                ).configure(
                    background="transparent" # Make chart bg transparent
                )
                
                st.altair_chart(chart)
                
                # Add download button for the performance data
                csv = perf_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Performance CSV", data=csv, file_name="model_performance.csv", mime='text/csv')

# --- TOP-RIGHT: Forecast (Line Chart) ---
with row1_col2:
    with st.container(border=True):
        c2_col1, c2_col2 = st.columns(2)
        with c2_col1:
            st.markdown("<h3>Forecast Comparison</h3>", unsafe_allow_html=True)
        
        if perf_df.empty:
            st.info("Select pollutant(s) and model(s) to run analysis.")
        else:
            with c2_col2:
                if not selected_pollutants_cache:
                    st.info("No pollutant selected.")
                    plot_pollutant = None
                else:
                    # Select which pollutant to display in the chart
                    plot_pollutant = st.selectbox("Pollutant", options=selected_pollutants_cache, format_func=lambda x: x.upper(), key='plot_poll_1')
            
            if plot_pollutant and plot_pollutant in all_plot_data:
                # Combine 'Actuals' and all model 'Predictions' into one dataframe
                plot_df = pd.concat(all_plot_data[plot_pollutant])
                plot_df['ds'] = pd.to_datetime(plot_df['ds'])
                
                # Base layer: The 'Actuals' line
                base = alt.Chart(plot_df.query("Model == 'Actuals'")).mark_line(color='#1c64f2').encode(
                    x=alt.X('ds:T', title='Date'),
                    y=alt.Y('Actuals:Q', title=plot_pollutant.upper()),
                    tooltip=[alt.Tooltip('ds:T', title='Date'), alt.Tooltip('Actuals:Q', format=".2f", title="Value")]
                )
                # Prediction layer: Dashed lines for each model
                preds_chart = alt.Chart(plot_df.query("Model != 'Actuals'")).mark_line(strokeDash=[5,5]).encode(
                    x=alt.X('ds:T'),
                    y=alt.Y('Actuals:Q'),
                    color=alt.Color('Model:N', legend=alt.Legend(title="Model", orient="bottom")),
                    tooltip=[alt.Tooltip('ds:T', title='Date'), alt.Tooltip('Model:N'), alt.Tooltip('Actuals:Q', format=".2f", title="Pred")]
                )
                
                # Combine the layers
                final_chart = (base + preds_chart).properties(height=300).configure_view(
                    strokeOpacity=0
                ).configure(
                    background="transparent" 
                )

                st.altair_chart(final_chart, use_container_width=True)

                # Download button for the plot data
                download_df = plot_df.copy()
                download_csv = download_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Forecast Data", data=download_csv, file_name=f"{plot_pollutant}_forecast.csv", mime='text/csv')

# --- BOTTOM-LEFT: Best Models (Table) ---
with row2_col1:
    with st.container(border=True):
        st.markdown("<h3>Best Model by Pollutant</h3>", unsafe_allow_html=True)
        
        if perf_df.empty:
            st.info("Select pollutant(s) and model(s) to run analysis.")
        else:
            best_models_list = []
            # Find the best model (lowest RMSE) for each pollutant
            for pollutant in selected_pollutants_cache:
                pollutant_df = perf_df[perf_df['Pollutant'] == pollutant.upper()]
                if not pollutant_df.empty:
                    # Find the row with the minimum RMSE
                    best_row = pollutant_df.loc[pollutant_df['RMSE'].idxmin()]
                    best_models_list.append({
                        'Pollutant': pollutant.upper(),
                        'Best Model': best_row['Model'],
                        'RMSE': f"{best_row['RMSE']:.2f}",
                        'Status': 'Active' # Just for styling
                    })
            best_model_df = pd.DataFrame(best_models_list)
            
            if best_model_df.empty:
                st.info("No best-model data yet.")
            else:
                # Display as a clean dataframe
                st.dataframe(best_model_df.style.applymap(lambda v: 'color: green; font-weight: 600;' if v=='Active' else '', subset=['Status']), use_container_width=True, hide_index=True)

# --- BOTTOM-RIGHT: Forecast Accuracy (Line Chart over Horizon) ---
with row2_col2:
    with st.container(border=True):
        c4_col1, c4_col2 = st.columns(2)
        with c4_col1:
            st.markdown("<h3>Forecast Accuracy</h3>", unsafe_allow_html=True)

        if perf_df.empty:
            st.info("Select pollutant(s) and model(s) to run analysis.")
        else:
            with c4_col2:
                # Select pollutant for this chart
                acc_pollutant = st.selectbox("Pollutant", options=selected_pollutants_cache, format_func=lambda x: x.upper(), key='acc_poll_1')

            try:
                models_store = all_plot_data.get('_models', {})
                if acc_pollutant and acc_pollutant in models_store:
                    # Get the data and trained models for the selected pollutant
                    df_json = raw_df.to_json(date_format='iso', orient='split')
                    train_data, test_data, full_clean_data, *_ = preprocess_data_cached(df_json, selected_city, acc_pollutant)
                    trained = models_store[acc_pollutant]
                    horizons = [1, 3, 7, 14, 30] # Define the horizons to test
                    
                    with st.spinner(f"Calculating accuracy for {acc_pollutant}..."):
                        # This is a heavy computation, so run it on demand
                        acc_dict = compute_horizon_accuracy(train_data, full_clean_data, test_data, trained, horizons=horizons, lstm_lookback=lstm_lookback)
                    
                    acc_df = pd.DataFrame({'Horizon (days)': horizons})
                    for model_name, arr in acc_dict.items():
                        acc_df[model_name] = arr
                    
                    # Melt for Altair
                    melt_df = acc_df.melt('Horizon (days)', var_name='Model', value_name='RMSE')
                    
                    # Create the line chart
                    chart = alt.Chart(melt_df).mark_line(point=True).encode(
                        x=alt.X('Horizon (days):O', title='Forecast Horizon (days)'),
                        y=alt.Y('RMSE:Q', title='RMSE'),
                        color=alt.Color('Model:N', legend=alt.Legend(title="Model", orient="bottom")),
                        tooltip=['Horizon (days)', 'Model', alt.Tooltip('RMSE', format=".2f")]
                    ).properties(height=300).interactive().configure_view(
                        strokeOpacity=0
                    ).configure(
                        background="transparent"
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No accuracy data available for this pollutant.")
            except Exception as e:
                st.error(f"Could not compute forecast accuracy: {e}")

# --- Optional Prophet components ---
st.markdown("---")
if not perf_df.empty and "Prophet" in selected_models:
    try:
        show_components = st.checkbox("Show Prophet components for a pollutant", value=False)
        if show_components:
            comp_poll = st.selectbox("Pollutant (Prop_components)", options=selected_pollutants_cache, format_func=lambda x: x.upper(), key='prophet_comp_poll')
            
            # Re-load data for this pollutant
            df_json = raw_df.to_json(date_format='iso', orient='split')
            train_data, test_data, full_clean_data, *_ = preprocess_data_cached(df_json, selected_city, comp_poll)
            
            if train_data.empty:
                st.info("Not enough data for components.")
            else:
                # --- Set Matplotlib style (Forced Default) ---
                plt.style.use('default')

                with st.spinner(f"Generating Prophet components for {comp_poll}..."):
                    
                    # --- FIX #1 (Prophet Labeling) ---
                    # We need to re-train and predict on the *full* dataset to show components properly
                    model = train_prophet(train_data[['ds','y']])
                    future_df = full_clean_data[['ds']] 
                    forecast = model.predict(future_df)
                    # ---------------------------
                    
                    # --- Set fig background to be transparent to match theme ---
                    fig = model.plot_components(forecast)
                    fig.patch.set_facecolor('none') # Make main bg transparent
                    for ax in fig.axes:
                        ax.patch.set_facecolor('none') # Make subplot bg transparent
                        
                    st.pyplot(fig) # Display the matplotlib plot
    except Exception as e:
        st.error(f"Could not display Prophet components: {e}")

st.caption("Built with Prophet, ARIMA, and LSTM — enhancements: caching, quick-mode, improved UI & downloads. Data loaded from your local CSV (clean data).")
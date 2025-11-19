# app_enhanced.py
import os
import json
import hashlib
import io

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

# Models
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback

# ---------------------------
# Page config & CSS / Fonts
# ---------------------------
st.set_page_config(layout="wide", page_title="Air Quality Forecast Engine", page_icon="🌫️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background-color: #f6f9fc;
}
h1 { color: #0f172a; font-weight:700; }
h2 { color: #1e293b; }
.stButton>button { border-radius: 8px; }
[data-testid="stSidebar"] {
    background-color: #ffffff;
    padding: 16px;
    border-right: 1px solid #e6e9ef;
}
div[data-testid="stContainer"][border="true"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px;
    min-height: 300px;
    box-shadow: 0 6px 18px rgba(17,24,39,0.04);
}
[data-testid="stDataFrame"] thead th { background-color: #f3f4f6; color: #0f172a; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Utility helpers
# ---------------------------
def df_hash(df: pd.DataFrame) -> str:
    """Return a short hash for dataframe contents (json) to use as cache-key."""
    j = df.to_json(date_format='iso', orient='split')
    return hashlib.md5(j.encode('utf-8')).hexdigest()

class StreamlitCallback(Callback):
    def __init__(self, progress_bar, epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.epoch_count = epochs

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epoch_count
        try:
            self.progress_bar.progress(progress, text=f"LSTM Epoch {epoch+1}/{self.epoch_count}")
        except Exception:
            pass

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# ---------------------------
# Data loading & preprocessing
# ---------------------------
@st.cache_data
def load_raw_data(path="human_readable_air_quality_CITYWISE.csv"):
    """Load the city-wise cleaned CSV. If absent, return empty df."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
    return df

@st.cache_data
def get_city_list(_df):
    if _df.empty:
        return []
    return sorted(_df['location'].unique())

@st.cache_data
def preprocess_data_cached(df_json: str, city: str, pollutant_col: str):
    """Preprocess - accepts df as json string to be cache-friendly."""
    df = pd.read_json(df_json, convert_dates=['date'], orient='split')
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, 0, 0
    city_df = df[df['location'] == city].copy()
    if city_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, 0, 0
    city_df.set_index('date', inplace=True)
    if pollutant_col not in city_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, 0, 0
    data_series = city_df[pollutant_col].resample('D').mean()
    data_series = data_series.interpolate(method='time').ffill().bfill()
    clean_data = data_series.to_frame(name='y')
    clean_data['ds'] = clean_data.index
    clean_data.reset_index(drop=True, inplace=True)
    total_points = len(clean_data)
    split_index = int(total_points * 0.8)
    if split_index < 2:
        return pd.DataFrame(), pd.DataFrame(), clean_data, total_points, 0, max(0, total_points - split_index)
    train_data = clean_data.iloc[:split_index]
    test_data = clean_data.iloc[split_index:]
    return train_data, test_data, clean_data, total_points, len(train_data), len(test_data)

# ---------------------------
# Model training (cache by data hash)
# ---------------------------
@st.cache_resource
def _train_prophet_cache(train_json: str, seed: int = 0):
    """Internal cached function returning a fitted Prophet. Keyed by train_json."""
    df = pd.read_json(train_json, orient='split', convert_dates=['ds'])
    model = Prophet()
    model.fit(df)
    return model

def train_prophet(train_df: pd.DataFrame):
    if train_df.empty:
        return None
    train_json = train_df.to_json(date_format='iso', orient='split')
    return _train_prophet_cache(train_json)

@st.cache_resource
def _train_arima_cache(y_json: str):
    y = json.loads(y_json)
    ser = pd.Series(y)
    model = ARIMA(ser, order=(5, 1, 0))
    fitted = model.fit()
    return fitted

def train_arima(train_df: pd.DataFrame):
    if train_df.empty:
        return None
    y_json = json.dumps(train_df['y'].tolist())
    return _train_arima_cache(y_json)

@st.cache_resource
def _train_lstm_cache(scaled_json: str, look_back: int, epochs: int, seed: int = 0):
    """Cache by scaled data + look_back + epochs to avoid retrain on trivial UI changes."""
    X = np.array(json.loads(scaled_json))
    # Note: X here is scaled dataset (1D list). We'll reconstruct properly.
    X = np.array(X).reshape(-1, 1)
    X_train, y_train = create_dataset(X, look_back)
    if X_train.shape[0] == 0:
        return None, None
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    # We can't show progress bar from cached function. Train for epochs but minimal.
    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
    # return model weights as bytes? Streamlit can cache Keras model object as resource
    # but Keras models are generally cacheable if returned.
    return model, None

def train_lstm(train_df: pd.DataFrame, look_back=30, epochs=20, quick_mode=False):
    if train_df.empty:
        return None, None
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(train_df['y'].values.reshape(-1, 1)).flatten().tolist()
    scaled_json = json.dumps(scaled)
    # If quick_mode, reduce epochs/look_back but still use caching key
    e = max(1, int(epochs))
    lb = max(1, int(look_back))
    # Train: use cached routine (which trains silently)
    model, _ = _train_lstm_cache(scaled_json, lb, e)
    return model, scaler

# ---------------------------
# Main app
# ---------------------------
raw_df = load_raw_data()
if raw_df.empty:
    st.title("Air Quality Forecast Engine")
    st.error("`human_readable_air_quality_CITYWISE.csv` not found in the app folder. Please upload your cleaned CSV with columns including: date, location, avg_rspm/avg_so2/avg_no2.")
    st.stop()

cities = get_city_list(raw_df)
pollutant_options = [c for c in ["avg_rspm", "avg_so2", "avg_no2"] if c in raw_df.columns]
if not pollutant_options:
    st.error("No pollutant columns found in CSV. Expected one of: avg_rspm, avg_so2, avg_no2.")
    st.stop()

model_options = ["Prophet", "ARIMA", "LSTM"]

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Forecast Controls")
    default_index = 0
    if "Hyderabad" in cities:
        default_index = cities.index("Hyderabad")
    selected_city = st.selectbox("Select City", options=cities, index=default_index)

    with st.expander("Pollutant Selection", expanded=True):
        selected_pollutants = st.multiselect("Select Pollutant(s)", options=pollutant_options, default=pollutant_options[:1])

    with st.expander("Model Selection", expanded=True):
        selected_models = st.multiselect("Select Model(s)", options=model_options, default=model_options)

    st.markdown("---")
    quick_mode = st.checkbox("⚡ Quick Mode (faster)", value=False)
    lstm_epochs = st.slider("LSTM Epochs", 1, 100, 20 if not quick_mode else 5)
    lstm_lookback = st.number_input("LSTM Look-back (days)", min_value=1, max_value=60, value=30 if not quick_mode else 15)
    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Initialize session state
if 'perf_df' not in st.session_state:
    st.session_state.perf_df = pd.DataFrame()
if 'all_plot_data' not in st.session_state:
    st.session_state.all_plot_data = {}
if 'data_info' not in st.session_state:
    st.session_state.data_info = {}
if 'selected_pollutants_cache' not in st.session_state:
    st.session_state.selected_pollutants_cache = []

st.title("Air Quality Forecast Engine")
st.subheader(f"Model Comparison for {selected_city}")

# Header metrics placeholder (will update after run)
metrics_placeholder = st.container()

# Run analysis
if run_analysis:
    if not selected_pollutants:
        st.error("Please select at least one pollutant.")
    elif not selected_models:
        st.error("Please select at least one model.")
    else:
        with st.spinner(f"Running analysis for {len(selected_pollutants)} pollutant(s) and {len(selected_models)} model(s)..."):
            all_model_results = []
            all_plot_data = {}
            data_info = {}
            st.session_state.selected_pollutants_cache = selected_pollutants

            df_json = raw_df.to_json(date_format='iso', orient='split')

            for pollutant in selected_pollutants:
                train_data, test_data, full_clean_data, total_pts, train_pts, test_pts = preprocess_data_cached(df_json, selected_city, pollutant)
                data_info[pollutant] = {"Total": total_pts, "Train": train_pts, "Test": test_pts}
                if train_data.empty or test_data.empty or total_pts < 5:
                    st.warning(f"Not enough data for {pollutant}. Skipping.")
                    continue

                actuals = test_data['y'].values
                # store actuals frame (full series)
                actuals_df = full_clean_data.rename(columns={'y': 'Actuals'})
                actuals_df['Model'] = 'Actuals'
                all_plot_data[pollutant] = [actuals_df]

                for model_name in selected_models:
                    try:
                        preds = None
                        if model_name == "Prophet":
                            model = train_prophet(train_data[['ds', 'y']])
                            test_future_df = pd.DataFrame({'ds': test_data['ds']})
                            forecast = model.predict(test_future_df)
                            preds = forecast['yhat'].values

                        elif model_name == "ARIMA":
                            model = train_arima(train_data[['y']])
                            preds = model.forecast(steps=len(test_data))

                        elif model_name == "LSTM":
                            # train LSTM (may be quick_mode)
                            model, scaler = train_lstm(train_data[['y']], look_back=lstm_lookback, epochs=lstm_epochs, quick_mode=quick_mode)
                            if model:
                                # Prepare inputs from full data anchored to test
                                inputs = full_clean_data['y'].values
                                # need enough look_back + test length
                                start_idx = len(inputs) - len(test_data) - lstm_lookback
                                if start_idx < 0:
                                    st.warning(f"LSTM for {pollutant} skipped: not enough historical points for look_back.")
                                    continue
                                inputs_seq = inputs[start_idx:].reshape(-1, 1)
                                scaler_for_inv = MinMaxScaler(feature_range=(0, 1))
                                scaler_for_inv.fit(full_clean_data['y'].values.reshape(-1,1))
                                inputs_scaled = scaler_for_inv.transform(inputs_seq)
                                X_test = []
                                for i in range(lstm_lookback, len(inputs_scaled)):
                                    X_test.append(inputs_scaled[i-lstm_lookback:i, 0])
                                X_test = np.array(X_test)
                                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                                test_preds_scaled = model.predict(X_test, verbose=0)
                                preds = scaler_for_inv.inverse_transform(test_preds_scaled).flatten()
                            else:
                                st.warning(f"LSTM for {pollutant} skipped: not enough data to form sequences.")
                                continue

                        # compute metrics
                        if preds is not None:
                            pred_len = len(preds)
                            actuals_sliced = actuals[:pred_len]
                            rmse = np.sqrt(mean_squared_error(actuals_sliced, preds))
                            mae = mean_absolute_error(actuals_sliced, preds)
                            all_model_results.append({
                                'Pollutant': pollutant.upper(),
                                'Model': model_name,
                                'RMSE': float(rmse),
                                'MAE': float(mae)
                            })

                            preds_df = pd.DataFrame({
                                'ds': test_data['ds'].values[:pred_len],
                                'Actuals': preds,
                                'Model': model_name
                            })
                            all_plot_data[pollutant].append(preds_df)
                    except Exception as e:
                        st.error(f"Training for {model_name} on {pollutant} failed: {e}")

            st.success("Analysis complete!")

            if not all_model_results:
                st.session_state.perf_df = pd.DataFrame()
                st.session_state.all_plot_data = {}
                st.session_state.data_info = {}
            else:
                st.session_state.perf_df = pd.DataFrame(all_model_results)
                st.session_state.all_plot_data = all_plot_data
                st.session_state.data_info = data_info

# ---------------------------
# Results display (Tabs)
# ---------------------------
if st.session_state.perf_df.empty:
    st.info("Run analysis to see model results and forecasts.")
else:
    perf_df = st.session_state.perf_df
    all_plot_data = st.session_state.all_plot_data
    data_info = st.session_state.data_info
    selected_pollutants_cache = st.session_state.selected_pollutants_cache

    # Header metrics
    try:
        best_global = perf_df.loc[perf_df['RMSE'].idxmin()]
        with metrics_placeholder:
            col1, col2, col3 = st.columns([2,2,1])
            col1.metric("Best Overall Model", f"{best_global['Model']}")
            col2.metric("Lowest RMSE", f"{best_global['RMSE']:.2f}")
            col3.metric("Pollutants Analyzed", len(selected_pollutants_cache))
    except Exception:
        pass

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "📈 Forecast", "🏆 Best Models", "📘 Data Info"])

    # --- Performance Tab ---
    with tab1:
        st.subheader("Model Performance (Test Set)")
        metric_choice = st.radio("Metric to display", ["RMSE", "MAE"], horizontal=True)
        chart_df = perf_df.melt(id_vars=['Pollutant', 'Model'], value_vars=['RMSE', 'MAE'], var_name='Metric', value_name='Value')
        chart_df = chart_df[chart_df['Metric'] == metric_choice]
        if chart_df.empty:
            st.info("No performance data to show.")
        else:
            chart = alt.Chart(chart_df).mark_bar(cornerRadius=6).encode(
                x=alt.X('Model:N', title="Model"),
                y=alt.Y('Value:Q', title=metric_choice),
                color=alt.Color('Model:N', scale=alt.Scale(scheme='tableau10')),
                column=alt.Column('Pollutant:N', header=alt.Header(labelAngle=0, labelOrient='bottom')),
                tooltip=[alt.Tooltip('Pollutant:N'), alt.Tooltip('Model:N'), alt.Tooltip('Value:Q', format=".2f")]
            ).properties(height=320).configure_view(strokeOpacity=0)
            st.altair_chart(chart, use_container_width=True)

            # Download performance CSV
            csv = perf_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Performance CSV", data=csv, file_name="model_performance.csv", mime='text/csv')

    # --- Forecast Tab ---
    with tab2:
        st.subheader("Forecast Comparison")
        if not selected_pollutants_cache:
            st.info("No pollutant selected.")
        else:
            plot_pollutant = st.selectbox("Select pollutant to view", options=selected_pollutants_cache, format_func=lambda x: x.upper())
            if plot_pollutant in all_plot_data:
                plot_df = pd.concat(all_plot_data[plot_pollutant])
                # Ensure ds is datetime
                plot_df['ds'] = pd.to_datetime(plot_df['ds'])
                base = alt.Chart(plot_df.query("Model == 'Actuals'")).mark_line().encode(
                    x=alt.X('ds:T', title='Date'),
                    y=alt.Y('Actuals:Q', title=plot_pollutant.upper()),
                    tooltip=[alt.Tooltip('ds:T', title='Date'), alt.Tooltip('Actuals:Q', format=".2f", title="Value")]
                )
                preds_chart = alt.Chart(plot_df.query("Model != 'Actuals'")).mark_line(strokeDash=[5,5]).encode(
                    x=alt.X('ds:T'),
                    y=alt.Y('Actuals:Q'),
                    color=alt.Color('Model:N', legend=alt.Legend(title="Model")),
                    tooltip=[alt.Tooltip('ds:T', title='Date'), alt.Tooltip('Model:N'), alt.Tooltip('Actuals:Q', format=".2f", title="Pred")]
                )
                final_chart = (base + preds_chart).properties(height=420).interactive().configure_view(strokeOpacity=0)
                st.altair_chart(final_chart, use_container_width=True)

                # Offer download of forecast (concatenate preds)
                download_df = plot_df.copy()
                download_csv = download_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Forecast Data (this pollutant)", data=download_csv, file_name=f"{plot_pollutant}_forecast.csv", mime='text/csv')
            else:
                st.info("No forecast plots to display for this pollutant.")

    # --- Best Models Tab ---
    with tab3:
        st.subheader("Best Model (by RMSE per pollutant)")
        best_models_list = []
        for pollutant in selected_pollutants_cache:
            pollutant_df = perf_df[perf_df['Pollutant'] == pollutant.upper()]
            if not pollutant_df.empty:
                best_row = pollutant_df.loc[pollutant_df['RMSE'].idxmin()]
                best_models_list.append({
                    'Pollutant': pollutant.upper(),
                    'Best Model': best_row['Model'],
                    'RMSE': f"{best_row['RMSE']:.2f}",
                    'Status': 'Active'
                })
        best_model_df = pd.DataFrame(best_models_list)
        if best_model_df.empty:
            st.info("No best-model data yet.")
        else:
            def color_status(val):
                color = 'green' if val == 'Active' else 'red'
                return f'color: {color}; font-weight: 600;'
            st.dataframe(best_model_df.style.applymap(lambda v: 'color: green' if v=='Active' else '', subset=['Status']), use_container_width=True, hide_index=True)

    # --- Data Info Tab ---
    with tab4:
        st.subheader("Data Info")
        if not selected_pollutants_cache:
            st.info("No pollutant selected.")
        else:
            info_pollutant = st.selectbox("Select pollutant", options=selected_pollutants_cache, format_func=lambda x: x.upper())
            if info_pollutant in data_info:
                info = data_info[info_pollutant]
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Cleaned Points", info['Total'])
                col2.metric("Training Points", info['Train'])
                col3.metric("Testing Points", info['Test'])
            else:
                st.info("No info for selected pollutant (maybe analysis not run).")

    # Optional: Prophet components viewer if Prophet was in selected models
    st.markdown("---")
    if "Prophet" in selected_models:
        try:
            show_components = st.checkbox("Show Prophet components for a pollutant", value=False)
            if show_components:
                comp_poll = st.selectbox("Pollutant (Prophet components)", options=selected_pollutants_cache, format_func=lambda x: x.upper())
                # Rebuild Prophet model on train split for that pollutant to produce components
                df_json = raw_df.to_json(date_format='iso', orient='split')
                train_data, test_data, full_clean_data, *_ = preprocess_data_cached(df_json, selected_city, comp_poll)
                if train_data.empty:
                    st.info("Not enough data for components.")
                else:
                    model = train_prophet(train_data[['ds','y']])
                    future = model.make_future_dataframe(periods=0)
                    forecast = model.predict(future)
                    fig = model.plot_components(forecast)
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not display Prophet components: {e}")

st.caption("Built with Prophet, ARIMA, and LSTM — enhancements: caching, quick-mode, improved UI & downloads. Data loaded from your local CSV (clean data).")

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Air Quality Alert System")

# ============================
# ===  CONFIG: set path  ====
# ============================
# EDIT THIS to your local CSV path:
DATA_PATH = r"D:\Internship\MILESTONE_1\Milesstone2\data.csv"
# Example: DATA_PATH = r"C:\Users\You\Downloads\data.csv"

# ============================
# ===  CPCB Breakpoints  ====
# ============================
# Breakpoint tables (µg/m3). Edit if you want different values.
BREAKPOINTS = {
    "pm2_5": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 350, 401, 500),
        (351, 99999, 501, 99999),
    ],
    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 500, 401, 500),
        (501, 99999, 501, 99999),
    ],
    "no2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 1000, 401, 500),
    ],
    "so2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2000, 401, 500),
    ],
    "o3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 99999, 401, 99999),
    ],
}

CATEGORY_LABELS = [
    (0, 50, "Good", "#2ecc71"),
    (51, 100, "Satisfactory", "#9be7a5"),
    (101, 200, "Moderate", "#ffb020"),
    (201, 300, "Poor", "#ff7a59"),
    (301, 400, "Very Poor", "#b347ff"),
    (401, 500, "Severe", "#7e0023"),
]

PRETTY = {"pm2_5": "PM2.5", "pm10": "PM10 (rspm)", "so2": "SO₂", "no2": "NO₂", "o3": "O₃"}

# ============================
# ===  helper functions  ====
# ============================
def sub_index(pollutant, conc):
    if conc is None or np.isnan(conc):
        return np.nan
    table = BREAKPOINTS.get(pollutant)
    if table is None:
        return np.nan
    for (bp_low, bp_high, i_low, i_high) in table:
        if bp_low <= conc <= bp_high:
            if bp_high == bp_low:
                return float(i_high)
            return (i_high - i_low) / (bp_high - bp_low) * (conc - bp_low) + i_low
    return np.nan

def aqi_category_and_color(aqi):
    if aqi is None or np.isnan(aqi):
        return "Unknown", "#dddddd"
    for low, high, label, color in CATEGORY_LABELS:
        if low <= aqi <= high:
            return label, color
    # >500 hazardous
    if aqi > 500:
        return "Hazardous", "#4b0000"
    return "Unknown", "#dddddd"

# ============================
# ===  load & prepare CSV  ==
# ============================
@st.cache_data(ttl=600)
def load_and_prepare(path):
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except Exception:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)

    # strip column names
    df.columns = [c.strip() for c in df.columns]

    # Rename common variants
    renames = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("pm2_5","pm2.5","pm25","pm2"):
            renames[c] = "pm2_5"
        if lc in ("pm10","rspm","r.s.p.m","rsm"):
            renames[c] = "rspm"
        if lc in ("so2","so_2","sulphur_dioxide"):
            renames[c] = "so2"
        if lc in ("no2","nitrogen_dioxide"):
            renames[c] = "no2"
        if lc in ("spm","tsp"):
            renames[c] = "spm"
        if lc in ("o3","ozone"):
            renames[c] = "o3"
        if lc in ("sampling_date",):
            renames[c] = "sampling_date"
        if lc in ("date",):
            renames[c] = "date"
        if lc in ("state",):
            renames[c] = "state"
        if lc in ("location",):
            renames[c] = "location"
    df = df.rename(columns=renames)

    # parse date: prefer 'date' then 'sampling_date'
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        if "sampling_date" in df.columns:
            df["date"] = pd.to_datetime(df["sampling_date"], errors="coerce")
        else:
            # try to find any column with date-like values
            for c in df.columns:
                if df[c].dtype == object and df[c].str.match(r"\d{4}-\d{2}-\d{2}").any():
                    df["date"] = pd.to_datetime(df[c], errors="coerce")
                    break

    # Map rspm to pm10
    if "rspm" in df.columns:
        df["pm10"] = df["rspm"]
    else:
        if "spm" in df.columns:
            df["pm10"] = df["spm"]

    # Ensure pollutant columns exist
    pollutants = ["pm2_5", "pm10", "so2", "no2", "o3"]
    for p in pollutants:
        if p not in df.columns:
            df[p] = np.nan

    # Drop rows where all pollutant values missing
    df = df.dropna(subset=pollutants, how="all").copy()

    # Fill missing pollutant values with column mean
    for p in pollutants:
        if df[p].notna().any():
            df[p] = df[p].fillna(df[p].mean(skipna=True))

    # Compute CPCB sub-indices and AQI (max of sub-indices)
    def compute_aqi_row(r):
        subs = []
        for p in ["pm2_5", "pm10", "so2", "no2", "o3"]:
            val = r.get(p, np.nan)
            if not pd.isna(val):
                subs.append(sub_index(p, float(val)))
        subs = [s for s in subs if not pd.isna(s)]
        return max(subs) if subs else np.nan

    df["AQI"] = df.apply(compute_aqi_row, axis=1)
    df["AQI_cat"], df["AQI_color"] = zip(*df["AQI"].apply(lambda x: aqi_category_and_color(x)))

    return df

# Attempt load
try:
    df = load_and_prepare(DATA_PATH)
except FileNotFoundError as e:
    st.error(f"Data file not found at: {DATA_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if df.empty:
    st.error("No pollutant data after cleaning. Check CSV contents.")
    st.stop()

# ============================
# ===  Sidebar Controls  ====
# ============================
st.sidebar.header("Filters")

# pollutants available
available_pollutants = [p for p in ["pm2_5", "pm10", "so2", "no2", "o3"] if p in df.columns]
poll_default = [p for p in ["pm2_5", "pm10", "so2", "no2"] if p in available_pollutants]
selected_pollutants = st.sidebar.multiselect("Select pollutants for AQI", available_pollutants, default=poll_default, format_func=lambda x: PRETTY.get(x, x))

states = sorted(df['state'].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("State", options=["All"] + states, index=0)

if selected_state != "All":
    locations = sorted(df.loc[df['state'] == selected_state, 'location'].dropna().unique().tolist())
else:
    locations = sorted(df['location'].dropna().unique().tolist())

selected_location = st.sidebar.selectbox("Location", options=["All"] + locations, index=0)

# date range
min_date = df['date'].min()
max_date = df['date'].max()
date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())

# ============================
# ===  Apply filters  =======
# ============================
fdf = df.copy()
if selected_state != "All":
    fdf = fdf[fdf['state'] == selected_state]
if selected_location != "All":
    fdf = fdf[fdf['location'] == selected_location]

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
fdf = fdf[(fdf['date'] >= start_date) & (fdf['date'] <= end_date)]
if fdf.empty:
    st.warning("No data for the selected filters/date range.")
    st.stop()

# If user picks pollutants not in BREAKPOINTS, we fallback to normalized average
unsupported = [p for p in selected_pollutants if p not in BREAKPOINTS]
use_fallback = False
if unsupported:
    use_fallback = True
    st.sidebar.info("Selected pollutant(s) missing CPCB breakpoints — using normalized-average fallback for AQI.")

def compute_aqi_selected(dfin, pollutants, fallback=False):
    dfc = dfin.copy()
    if fallback:
        # normalize each pollutant to 0-500 using observed max (simple)
        norms = []
        for p in pollutants:
            maxv = dfc[p].max() if p in dfc.columns else 1
            pnorm = dfc[p] / maxv * 500 if maxv and maxv>0 else 0
            dfc[p + "_norm"] = pnorm
            norms.append(p + "_norm")
        dfc["AQI_calc"] = dfc[norms].mean(axis=1)
    else:
        def row_aqi(r):
            subvals = []
            for p in pollutants:
                if p in BREAKPOINTS:
                    val = r.get(p, np.nan)
                    if not pd.isna(val):
                        subvals.append(sub_index(p, float(val)))
            subvals = [s for s in subvals if not pd.isna(s)]
            return max(subvals) if subvals else np.nan
        dfc["AQI_calc"] = dfc.apply(row_aqi, axis=1)
    dfc["AQI_calc_cat"], dfc["AQI_calc_color"] = zip(*dfc["AQI_calc"].apply(lambda x: aqi_category_and_color(x)))
    return dfc

fdf = compute_aqi_selected(fdf, selected_pollutants, fallback=use_fallback)

# ============================
# ===  Styling (match design) =
# ============================
st.markdown(
    """
    <style>
    /* Page background */
    .stApp { background: #fffdfb; }
    .hero {
       
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 18px;
        background-color: #fa9b6b;
    }
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }
    .muted { color: #666; font-size: 0.95rem; }
    .forecast-box { border-radius:10px; padding:10px; text-align:center; color:#fff; font-weight:700; }
    .small-muted { color:#999; font-size:0.85rem; }
    </style>
    """, unsafe_allow_html=True
)

# Header / Hero
st.markdown("<div class='hero'><h1 style='color:#e96900; margin:0'>Air Quality Alert System</h1><div class='muted'>Milestone 3: Working Application</div></div>", unsafe_allow_html=True)

# Top layout: donut (left) and 7-day forecast (right)
col1, col2 = st.columns([1.2, 1])

# Donut
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Current Air Quality")
    latest = fdf.sort_values("date").iloc[-1]
    aqi_latest = latest["AQI_calc"]
    cat, color = aqi_category_and_color(aqi_latest)
    if np.isnan(aqi_latest):
        st.info("AQI not available for selected pollutants.")
    else:
        fig = go.Figure(go.Pie(
            values=[aqi_latest, max(0, 500 - aqi_latest)],
            labels=[f"AQI {int(round(aqi_latest))}", ""],
            hole=0.62,
            marker=dict(colors=[color, "#f5f5f5"]),
            sort=False, textinfo='none', hoverinfo='none'
        ))
        fig.update_layout(margin=dict(t=0,b=0,l=0,r=0), height=330, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<div style='font-size:26px; font-weight:700; color:{color}; margin-top:8px'>{int(round(aqi_latest))}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>{cat} • {latest['date'].date()}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 7-day forecast as small boxes
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("7-Day Forecast")
    # Prepare daily mean AQI and 7-day rolling mean
    daily = fdf.groupby('date').agg({'AQI_calc': 'mean'}).reset_index().sort_values('date')
    if daily.empty:
        st.info("Not enough data for forecast.")
    else:
        daily['AQI_7day'] = daily['AQI_calc'].rolling(7, min_periods=1).mean()
        last_date = daily['date'].max()
        # take last 7 values and use as simple projection for next 7 days
        last_vals = daily.tail(7)['AQI_7day'].values
        if len(last_vals) < 7:
            last_vals = np.pad(last_vals, (0, 7 - len(last_vals)), 'edge')
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        cols = st.columns(7)
        for i in range(7):
            d = future_dates[i]
            a = last_vals[i]
            lbl, colr = aqi_category_and_color(a)
            with cols[i]:
                st.markdown(f"<div class='forecast-box' style='background:{colr};'>{d.strftime('%a')}<div style='font-size:14px'>{int(round(a))}</div><div style='font-size:12px'>{lbl}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Second row: pollutant line chart left, active alerts right
col3, col4 = st.columns([1.6, 1])

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Pollutant Concentrations")
    to_plot = [p for p in selected_pollutants if p in fdf.columns]
    if not to_plot:
        st.info("No pollutant columns selected.")
    else:
        plot_df = fdf.set_index('date').sort_index()
        daily_plot = plot_df[to_plot].resample('D').mean().reset_index()
        fig2 = go.Figure()
        for p in to_plot:
            fig2.add_trace(go.Scatter(x=daily_plot['date'], y=daily_plot[p], name=PRETTY.get(p,p), mode='lines+markers'))
        fig2.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10), legend=dict(orientation="h"))
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Active Alerts")
    aqi_now = fdf.sort_values('date').iloc[-1]['AQI_calc']
    if np.isnan(aqi_now):
        st.write("AQI unavailable.")
    else:
        alerts = []
        if aqi_now >= 401:
            alerts.append(("Severe — avoid all outdoor exposure", "#7e0023"))
        elif aqi_now >= 301:
            alerts.append(("Very Poor — vulnerable groups avoid outdoor exposure", "#b347ff"))
        elif aqi_now >= 201:
            alerts.append(("Poor — reduce prolonged outdoor exertion", "#ff7a59"))
        elif aqi_now >= 101:
            alerts.append(("Moderate — sensitive groups take precautions", "#ffb020"))
        else:
            alerts.append(("Good / Satisfactory", "#2ecc71"))

        # detect sharp rise vs previous day
        daily = fdf.groupby('date').agg({'AQI_calc': 'mean'}).reset_index().sort_values('date')
        if len(daily) >= 2 and daily['AQI_calc'].iloc[-1] - daily['AQI_calc'].iloc[-2] > 50:
            alerts.append(("Sharp rise in AQI since yesterday", "#ff7a59"))

        for msg, bg in alerts:
            st.markdown(f"<div style='background:{bg}; color:white; padding:12px; border-radius:8px; margin-bottom:10px'>{msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Bottom: data & summary
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Data & Summary")
st.markdown(f"**Records:** {len(fdf)} • **Range:** {fdf['date'].min().date()} — {fdf['date'].max().date()}")

c1, c2, c3 = st.columns(3)
latest_row = fdf.sort_values('date').iloc[-1]
with c1:
    st.metric("Latest AQI", int(round(latest_row["AQI_calc"])) if not np.isnan(latest_row["AQI_calc"]) else "n/a", latest_row["AQI_calc_cat"])
with c2:
    st.metric("Avg AQI (range)", f"{fdf['AQI_calc'].mean():.1f}")
with c3:
    st.metric("Max AQI (range)", int(round(fdf['AQI_calc'].max())))

# show dataframe (selected cols)
show_cols = ["date", "state", "location"] + [p for p in selected_pollutants if p in fdf.columns] + ["AQI_calc", "AQI_calc_cat"]
st.dataframe(fdf.sort_values('date', ascending=False).reset_index(drop=True).loc[:, show_cols], height=300)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="margin-top:12px; color:#666; font-size:0.9rem">
    Notes: <strong>rspm</strong> is treated as <strong>PM10</strong>. Units are assumed µg/m³.
    CPCB breakpoints are used for AQI calculation. If selected pollutants lack breakpoints,
    a normalized-average fallback is used.
    </div>
    """, unsafe_allow_html=True
)

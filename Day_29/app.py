import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# ==========================
# PAGE HEADER
# ==========================
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #333333;
    }
    .subtitle {
        text-align: center;
        color: #555555;
        font-size: 18px;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🌫️ Air Quality Monitoring Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Visualize air pollution trends and city-wise statistics</div>', unsafe_allow_html=True)

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv("air_quality_data.csv")  # Make sure this file is in same directory

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Detect datetime column
    datetime_col = None
    for col in df.columns:
        if "date" in col or "time" in col:
            datetime_col = col
            break

    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        df = df.dropna(subset=[datetime_col])
        df = df.rename(columns={datetime_col: "datetime"})
    else:
        df["datetime"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")

    # Detect city column
    city_col = None
    for col in df.columns:
        if "city" in col or "location" in col:
            city_col = col
            break

    if city_col:
        df = df.rename(columns={city_col: "city"})
    else:
        df["city"] = "Unknown"

    # Ensure numeric columns are converted
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    return df

df = load_data()

# ==========================
# SIDEBAR FILTERS
# ==========================
st.sidebar.header("🔍 Filters")

cities = df["city"].unique()
selected_city = st.sidebar.selectbox("Select City", cities)

# Only keep numeric pollutant columns
pollutant_cols = [
    col for col in df.columns
    if col not in ["datetime", "city"]
    and pd.api.types.is_numeric_dtype(df[col])
]

if not pollutant_cols:
    st.error("No numeric pollutant columns found in the dataset!")
    st.stop()

selected_pollutants = st.sidebar.multiselect(
    "Select Pollutants", pollutant_cols, default=pollutant_cols[:3]
)

# Filter data
filtered_df = df[df["city"] == selected_city]

# ==========================
# KPI METRICS
# ==========================
st.markdown("### 📊 Key Pollution Metrics")

col1, col2, col3, col4 = st.columns(4)

for i, col in enumerate([col1, col2, col3, col4]):
    if i < len(selected_pollutants):
        pollutant = selected_pollutants[i]
        avg = filtered_df[pollutant].mean()
        max_val = filtered_df[pollutant].max()
        col.metric(f"{pollutant.upper()} (avg)", f"{avg:.2f}", delta=f"Max: {max_val:.2f}")

# ==========================
# TIME SERIES CHART
# ==========================
st.markdown("### ⏱️ Pollution Trend Over Time")

if selected_pollutants:
    fig1 = px.line(
        filtered_df,
        x="datetime",
        y=selected_pollutants,
        title=f"Pollution Levels Over Time ({selected_city})",
        template="plotly_white",
    )
    st.plotly_chart(fig1, use_container_width=True)

# ==========================
# CORRELATION ANALYSIS
# ==========================
st.markdown("### 🔗 Pollutant Correlation Analysis")

numeric_df = filtered_df[pollutant_cols].apply(pd.to_numeric, errors="coerce")
corr_df = numeric_df.corr().round(2)

corr_pairs = corr_df.unstack().reset_index()
corr_pairs.columns = ["Pollutant1", "Pollutant2", "Correlation"]

# Remove duplicates and self-pairs
corr_pairs = corr_pairs[corr_pairs["Pollutant1"] != corr_pairs["Pollutant2"]]
corr_pairs = corr_pairs.dropna(subset=["Correlation"])
corr_pairs["Size"] = np.abs(corr_pairs["Correlation"]) * 60

fig2 = px.scatter(
    corr_pairs,
    x="Pollutant1",
    y="Pollutant2",
    size="Size",
    color="Correlation",
    color_continuous_scale="greens",
    title="Pollutant Correlations",
)
fig2.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")

st.plotly_chart(fig2, use_container_width=True)

# ==========================
# CITY COMPARISON
# ==========================
st.markdown("### 🏙️ City-wise Average Pollution Levels")

city_avg = df.groupby("city")[pollutant_cols].mean().reset_index()

fig3 = px.bar(
    city_avg.melt(id_vars="city", var_name="Pollutant", value_name="Average Value"),
    x="city",
    y="Average Value",
    color="Pollutant",
    barmode="group",
    template="plotly_white",
    title="Average Pollution Levels by City",
)
st.plotly_chart(fig3, use_container_width=True)

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Built using Streamlit & Plotly</div>",
    unsafe_allow_html=True,
)

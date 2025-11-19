import streamlit as st
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
import datetime

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Air Quality Data Explorer")

# --- Custom CSS Styling ---
# This styles the app's colors, cards, and title.
st.markdown("""
<style>
/* ------------------- General Page Styling ------------------- */
.stApp {
    background-color: #f0f7f0; /* Light green background */
}

/* ------------------- Title Styling ------------------- */
.title-box {
    background-color: #f5f5f5;      
    border: 1px solid #e0e0e0;      
    border-radius: 10px;          
    padding: 20px 25px;    
    margin-bottom: 25px; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
}
.title-box h1 {
    color: #2E8B57;        
    font-weight: bold;
    font-size: 2.5rem;
    margin: 0;  
    text-align: left;
}
.title-box h3 {
    color: #2E8B57;
    font-weight: 600;
    margin-top: 8px;
    margin-bottom: 0;
}

/* ------------------- Subheaders (h3) Across App ------------------- */
/* Standardize card headers */
div[data-testid="stMarkdownContainer"] h3,
section.main h3,
h3 {
    color: #2E8B57 !important;    /* Force green color */
    font-weight: 600;
    margin-top: 0.8rem;
    margin-bottom: 0.6rem;
}

/* ------------------- Card Styling ------------------- */
/* Styles the `st.container(border=True)` */
div[data-testid="stContainer"][border="true"] {
    background-color: #ffffff; /* Plain white cards */
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 15px;
    height: 100%;
}

/* ------------------- Sidebar ------------------- */
[data-testid="stSidebar"] {
    background-color: #f8f9fa; /* Light gray sidebar */
}

/* ------------------- Metric Boxes ------------------- */
[data-testid="stMetric"] {
    background-color: #e8f5e9; /* Light green background */
    border-radius: 8px;
    padding: 15px;
    border: 1px solid #c8e6c9;
}

/* ------------------- Sidebar Buttons ------------------- */
[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background-color: #45a049;
    color: white;
}

/* ------------------- Progress Bars ------------------- */
[data-testid="stProgressBar"] > div {
    background-color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# --- Title Box (HTML) ---
# Display the main title using the custom HTML/CSS
st.markdown("""
<div class="title-box">
    <h1>Air Quality Data Explorer</h1>
    <h3>Milestone 1: Working Application (Weeks 1-2)</h3>
</div>
""", unsafe_allow_html=True)

# --- Data Loading Function ---
# Cache the data loading so it only runs once
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv", encoding='latin-1', low_memory=False)
        
        # --- Data Cleaning ---
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True) # Drop rows with unparseable dates

        # Get date range for sidebar info
        min_date = df['date'].min()
        max_date = df['date'].max()

        # Clean locations
        df['location'] = df['location'].fillna('Unknown')
        locations = sorted(df['location'].unique().tolist())

        # Convert pollutant columns to numeric, forcing errors to NaN
        pollutant_cols = ['so2', 'no2', 'rspm', 'pm2_5', 'spm']
        for col in pollutant_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Rename columns for clarity (e.g., rspm -> PM10)
        rename_map = {'so2': 'SO2', 'no2': 'NO2', 'rspm': 'PM10', 'pm2_5': 'PM2.5', 'spm': 'SPM'}
        df.rename(columns=rename_map, inplace=True)
        
        # Set date as index for time series analysis
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        # Find which pollutants are actually available in the cleaned data
        available_pollutants = [col for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'SPM'] if col in df.columns]
        
        return df, locations, available_pollutants, min_date, max_date
    
    except FileNotFoundError:
        st.error("Error: `data.csv` not found. Place it in the same directory as this app.")
        return pd.DataFrame(), [], [], None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), [], [], None, None

# --- Load Data ---
df, locations, available_pollutants, min_date, max_date = load_data()

# Stop the app if data loading failed
if df.empty:
    st.stop()

# --- CONSOLIDATED Sidebar Controls ---
with st.sidebar:
    st.header("Data Controls")
    st.divider()

    # Set default location to Hyderabad if it exists
    default_loc_index = locations.index("Hyderabad") if "Hyderabad" in locations else 0
    location = st.selectbox("Location", options=locations, index=default_loc_index)

    # Time range selector
    time_range = st.selectbox("Time Range", options=["Last 90 Days of Data", "Last Year of Data", "All Time"], index=1)

    # Main pollutant selector (for time series, stats, hist)
    # Set default to PM10 if it exists
    main_pollutant_index = available_pollutants.index('PM10') if 'PM10' in available_pollutants else 0
    main_pollutant = st.selectbox(
        "Select Pollutant (for Time Series, Stats, Distribution)",
        options=available_pollutants,
        index=main_pollutant_index
    )

    # Correlation pollutant selector
    default_corr = [p for p in ['PM2.5', 'PM10', 'NO2'] if p in available_pollutants]
    correlation_pollutants = st.multiselect(
        "Select Pollutants (for Correlation)",
        options=available_pollutants,
        default=default_corr
    )

    st.button("Apply Filters", type="primary", use_container_width=True)
    st.divider()

    # --- Sidebar Info Panels ---
    st.header("Data Quality")
    st.markdown("Completeness: **92%**") # Hardcoded example
    st.progress(92)
    st.markdown("Validity: **87%**") # Hardcoded example
    st.progress(87)

    # Show dataset date range
    if min_date and max_date:
        st.header("Dataset Info")
        st.info(f"Data available from:\n{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

# --- Data Filtering (Based on Sidebar) ---
# 1. Filter by Location
df_by_location = df[df['location'] == location].copy()

# 2. Filter by Time Range
if time_range == "Last 90 Days of Data":
    cutoff = max_date - DateOffset(days=90)
    df_filtered = df_by_location[df_by_location.index >= cutoff]
elif time_range == "Last Year of Data":
    cutoff = max_date - DateOffset(years=1)
    df_filtered = df_by_location[df_by_location.index >= cutoff]
else: # "All Time"
    df_filtered = df_by_location.copy()

# 3. Resample to daily average
# This smooths out the data and handles multiple readings per day
if not df_filtered.empty:
    df_filtered = df_filtered[available_pollutants].resample('D').mean().dropna(how='all')

# --- Main Dashboard Layout ---
col1, col2 = st.columns(2)

# --- Column 1 ---
with col1:
    # Card 1.1: Time Series Chart
    with st.container(border=True):
        st.subheader(f"{main_pollutant} Time Series for {location}")
        # Check if data exists for the selected pollutant
        if not df_filtered.empty and main_pollutant in df_filtered and not df_filtered[main_pollutant].isnull().all():
            ts_data = df_filtered[[main_pollutant]].dropna()
            st.line_chart(ts_data, color="#2E8B57")
        else:
            st.warning(f"No data found for {main_pollutant} in {location}.")

    # Card 1.2: Statistical Summary
    with st.container(border=True):
        st.subheader(f"{main_pollutant} Statistical Summary")
        if not df_filtered.empty and main_pollutant in df_filtered and not df_filtered[main_pollutant].isnull().all():
            pollutant_data = df_filtered[main_pollutant].dropna()
            
            # Calculate key stats
            summary_data = {
                'Mean': pollutant_data.mean(),
                'Median': pollutant_data.median(),
                'Max': pollutant_data.max(),
                'Min': pollutant_data.min(),
                'Std Dev': pollutant_data.std(),
                'Data Points': len(pollutant_data)
            }
            
            # Display stats in two columns of metrics
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric("Mean (µg/m³)", f"{summary_data['Mean']:.1f}")
                st.metric("Max (µg/m³)", f"{summary_data['Max']:.1f}")
                st.metric("Std Dev", f"{summary_data['Std Dev']:.1f}")
            with mcol2:
                st.metric("Median (µg/m³)", f"{summary_data['Median']:.1f}")
                st.metric("Min (µg/m³)", f"{summary_data['Min']:.1f}")
                st.metric("Data Points", f"{summary_data['Data Points']:,}")
        else:
            st.warning(f"No data to calculate stats for {main_pollutant}.")

# --- Column 2 ---
with col2:
    # Card 2.1: Pollutant Correlations
    with st.container(border=True):
        st.subheader(f"Pollutant Correlations in {location}")
        # Need at least 2 pollutants to show correlation
        if len(correlation_pollutants) < 2:
            st.info("Select at least two pollutants to see correlations.")
        elif not df_filtered.empty:
            corr_matrix = df_filtered[correlation_pollutants].corr()
            
            # "Melt" the matrix into a long format for the scatter plot
            corr_data = corr_matrix.stack().reset_index()
            corr_data.columns = ['Pollutant 1', 'Pollutant 2', 'Correlation']
            # Remove self-correlation (e.g., PM10 vs PM10)
            corr_data = corr_data[corr_data['Pollutant 1'] != corr_data['Pollutant 2']]
            # Use absolute correlation for dot size
            corr_data['Correlation Strength'] = corr_data['Correlation'].abs() * 100
            
            if not corr_data.empty:
                st.scatter_chart(corr_data, x='Pollutant 1', y='Pollutant 2',
                                 size='Correlation Strength', color="#4CAF50", height=300)
                # Manual legend
                st.markdown("**Legend:** <span style='color:#4CAF50;font-size:1.5em;'>●</span> "
                            "Size = Correlation Strength (Absolute Value)", unsafe_allow_html=True)
            else:
                st.warning("Could not calculate correlations.")
        else:
            st.warning("No data available for selected time range.")

    # Card 2.2: Distribution Analysis (Histogram)
    with st.container(border=True):
        st.subheader(f"{main_pollutant} Distribution Analysis")
        if not df_filtered.empty and main_pollutant in df_filtered and not df_filtered[main_pollutant].isnull().all():
            pollutant_data = df_filtered[main_pollutant].dropna()
            
            # Calculate histogram bins and values using numpy
            hist_values, bin_edges = np.histogram(pollutant_data, bins=10)
            # Create labels for the bins (e.g., "0-50")
            bin_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(len(bin_edges)-1)]
            # Format data for st.bar_chart
            hist_data = pd.DataFrame({'Frequency': hist_values, 'Range': bin_labels}).set_index('Range')
            
            st.bar_chart(hist_data, color="#4CAF50")
        else:
            st.warning(f"No data to build distribution for {main_pollutant}.")
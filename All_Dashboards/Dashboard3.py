import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide", page_title="Air Quality Alert System", page_icon="🌫️")

# ---------------- 1. CSS & STYLING ----------------
st.markdown(
    """
    <style>
    /* GLOBAL FONTS & BACKGROUND */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #fdfcfb 0%, #f7f8fc 40%, #eef1f6 100%);
        color: #1e293b;
    }

    /* REMOVE DEFAULT PADDING & HEADER */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* HERO SECTION */
    .hero-container {
        background: linear-gradient(120deg, rgba(255,245,235,0.8), rgba(255,255,255,0.9));
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(255,200,150,0.3);
        box-shadow: 0 10px 30px -10px rgba(255,150,100,0.15);
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 8px;
    }

    /* CARD CONTAINER */
    .custom-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #f1f5f9;
        height: 100%;
    }
    
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* PILLS (Forecast) - FORCE FLEX */
    .pill-container {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 10px !important;
        padding-bottom: 8px;
    }
    .pill {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 10px 14px;
        min-width: 70px;
        transition: transform 0.2s;
    }
    .pill:hover {
        transform: translateY(-2px);
        border-color: #cbd5e1;
    }
    .pill-day {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
    }
    .pill-val {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        margin: 4px 0;
    }
    .pill-lbl {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        color: #475569;
    }

    /* ALERT BOXES */
    .alert-box {
        display: flex !important;
        align-items: center !important;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        border-left: 4px solid;
        background-color: #fff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .alert-icon {
        font-size: 1.5rem;
        margin-right: 12px;
    }
    .alert-text {
        font-weight: 500;
        font-size: 0.95rem;
        color: #334155;
    }

    /* METRIC CARDS - FORCE GRID */
    .metric-grid {
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)) !important;
        gap: 16px !important;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True
)

# ---------------- CONFIG & DATA ----------------
DATA_PATH = "./data.csv"

BREAKPOINTS = {
    "pm2_5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 350, 401, 500), (351, 9999, 501, 9999)],
    "pm10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 500, 401, 500), (501, 9999, 501, 9999)],
    "no2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 1000, 401, 500)],
    "so2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500)],
    "o3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 9999, 401, 9999)],
}

CATEGORY_LABELS = [
    (0, 50, "Good", "#22c55e"),
    (51, 100, "Satisfactory", "#a3e635"),
    (101, 200, "Moderate", "#facc15"),
    (201, 300, "Poor", "#fb923c"),
    (301, 400, "Very Poor", "#c084fc"),
    (401, 500, "Severe", "#be123c"),
]

PRETTY = {"pm2_5": "PM2.5", "pm10": "PM10", "so2": "SO₂", "no2": "NO₂", "o3": "O₃"}

# ---------------- HELPERS ----------------
def sub_index(pollutant, conc):
    if conc is None or np.isnan(conc): return np.nan
    table = BREAKPOINTS.get(pollutant)
    if not table: return np.nan
    for (bp_low, bp_high, i_low, i_high) in table:
        if bp_low <= conc <= bp_high:
            return (i_high - i_low) / (bp_high - bp_low) * (conc - bp_low) + i_low
    return np.nan

def aqi_category_and_color(aqi):
    if aqi is None or np.isnan(aqi): return "Unknown", "#e2e8f0"
    for low, high, label, color in CATEGORY_LABELS:
        if low <= aqi <= high: return label, color
    if aqi > 500: return "Hazardous", "#881337"
    return "Unknown", "#e2e8f0"

@st.cache_data(ttl=600)
def load_data(path):
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
    
    # 1. Standardize basic column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # 2. Define renames
    renames = {
        "pm2.5": "pm2_5", "pm25": "pm2_5", "pm2": "pm2_5",
        "rspm": "pm10", "spm": "pm10", "r.s.p.m": "pm10", "pm10": "pm10",
        "sulphur_dioxide": "so2", "so2": "so2", "so_2": "so2",
        "nitrogen_dioxide": "no2", "no2": "no2",
        "ozone": "o3", "o3": "o3",
        "sampling_date": "date", "date": "date",
        "state": "state", "location": "location"
    }
    
    # 3. Safe Rename Strategy
    df.rename(columns=renames, inplace=True)
    
    # 4. Remove Duplicates caused by renaming
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Date parsing
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors='coerce', format='mixed')
    
    # Ensure columns exist
    for p in ["pm2_5", "pm10", "so2", "no2", "o3"]:
        if p not in df.columns: df[p] = np.nan
        else: df[p] = pd.to_numeric(df[p], errors='coerce')

    # Drop empty rows
    df = df.dropna(subset=["date"]).copy()
    
    # Simple imputation (mean)
    for p in ["pm2_5", "pm10", "so2", "no2", "o3"]:
        if df[p].notna().any():
            df[p] = df[p].fillna(df[p].mean())

    return df

# ---------------- LOAD DATA ----------------
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1163/1163624.png", width=50)
st.sidebar.title("Configuration")

# Filter: Pollutants
available_p = [p for p in ["pm2_5", "pm10", "so2", "no2", "o3"] if p in df.columns]
selected_p = st.sidebar.multiselect("Select Pollutants", available_p, default=available_p[:2], format_func=lambda x: PRETTY.get(x,x))

# Filter: State/Location
if "state" in df.columns:
    states = ["All"] + sorted(df['state'].dropna().unique().tolist())
    sel_state = st.sidebar.selectbox("State", states)
else: sel_state = "All"

if "location" in df.columns:
    if sel_state != "All":
        locs = ["All"] + sorted(df[df['state'] == sel_state]['location'].dropna().unique().tolist())
    else:
        locs = ["All"] + sorted(df['location'].dropna().unique().tolist())
    sel_loc = st.sidebar.selectbox("Location", locs)
else: sel_loc = "All"

# Filter: Date
min_d, max_d = df['date'].min().date(), df['date'].max().date()
dates = st.sidebar.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

# Export Button
st.sidebar.markdown("---")
st.sidebar.download_button("Download Full CSV", df.to_csv(index=False), "aqi_full.csv", "text/csv")

# ---------------- FILTER LOGIC ----------------
fdf = df.copy()
if sel_state != "All": fdf = fdf[fdf['state'] == sel_state]
if sel_loc != "All": fdf = fdf[fdf['location'] == sel_loc]
if isinstance(dates, tuple) and len(dates) == 2:
    fdf = fdf[(fdf['date'].dt.date >= dates[0]) & (fdf['date'].dt.date <= dates[1])]

if fdf.empty:
    st.warning("No data found for selected filters.")
    st.stop()

# Calculate AQI for filtered data
def calc_aqi(row):
    vals = []
    for p in selected_p:
        v = sub_index(p, row.get(p))
        if not pd.isna(v): vals.append(v)
    return max(vals) if vals else np.nan

fdf['AQI_calc'] = fdf.apply(calc_aqi, axis=1)
latest_row = fdf.sort_values('date').iloc[-1]
current_aqi = latest_row['AQI_calc']
curr_cat, curr_col = aqi_category_and_color(current_aqi)

# ---------------- LAYOUT: HEADER ----------------
st.markdown(f"""
    <div class="hero-container">
        <h1 class="hero-title">Air Quality Alert System</h1>
        <div class="hero-subtitle">Real-time monitoring & alerts for {sel_loc if sel_loc != 'All' else 'All Locations'}</div>
    </div>
""", unsafe_allow_html=True)

# ---------------- LAYOUT: TOP ROW (Gauge + Forecast) ----------------
c1, c2 = st.columns([1.2, 1.8])

with c1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>☁️</span> Current Air Quality</div>', unsafe_allow_html=True)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_aqi if not pd.isna(current_aqi) else 0,
        number={'font': {'size': 40, 'color': curr_col}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#ccc"},
            'bar': {'color': curr_col},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 50], 'color': '#dcfce7'},
                {'range': [50, 100], 'color': '#ecfccb'},
                {'range': [100, 200], 'color': '#fef9c3'},
                {'range': [200, 300], 'color': '#ffedd5'},
                {'range': [300, 400], 'color': '#f3e8ff'},
                {'range': [400, 500], 'color': '#ffe4e6'},
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<div style='text-align:center; color:#64748b; font-size:0.9rem'>Status: <strong>{curr_cat}</strong></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>📅</span> 7-Day Forecast Trend</div>', unsafe_allow_html=True)
    
    daily = fdf.groupby('date')['AQI_calc'].mean().reset_index().sort_values('date')
    if not daily.empty:
        daily['rolling'] = daily['AQI_calc'].rolling(7, min_periods=1).mean()
        last_7 = daily.tail(7)
        
        pills_html = '<div class="pill-container">'
        
        for idx, row in last_7.iterrows():
            val = row['rolling']
            d_str = row['date'].strftime("%a")
            c_lbl, c_col = aqi_category_and_color(val)
            bg_col = c_col + "20"
            pills_html += f'<div class="pill" style="background-color:{bg_col}; border-color:{c_col}40;">'
            pills_html += f'<span class="pill-day">{d_str}</span>'
            pills_html += f'<span class="pill-val">{val:.0f}</span>'
            pills_html += f'<span class="pill-lbl" style="color:{c_col}; background:rgba(255,255,255,0.5)">{c_lbl}</span>'
            pills_html += '</div>'

        start_forecast = daily['date'].max() + timedelta(days=1)
        remaining = 7 - len(last_7)
        last_val = daily['rolling'].iloc[-1] if not daily.empty else 0
        for i in range(remaining):
            f_date = start_forecast + timedelta(days=i)
            c_lbl, c_col = aqi_category_and_color(last_val)
            bg_col = c_col + "20"
            pills_html += f'<div class="pill" style="background-color:{bg_col}; border-color:{c_col}40; opacity:0.7;">'
            pills_html += f'<span class="pill-day">{f_date.strftime("%a")}</span>'
            pills_html += f'<span class="pill-val">{last_val:.0f}</span>'
            pills_html += f'<span class="pill-lbl">FCST</span>'
            pills_html += '</div>'
            
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)
        st.info("Forecast based on 7-day rolling average of historical data.")
    else:
        st.write("Not enough data for forecast.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- LAYOUT: MIDDLE ROW (Charts + Alerts) ----------------
c3, c4 = st.columns([2, 1])

with c3:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>📈</span> Pollutant Trends</div>', unsafe_allow_html=True)
    
    chart_df = fdf.groupby('date')[selected_p].mean().reset_index()
    fig2 = go.Figure()
    colors_map = {"pm2_5":"#ef4444", "pm10":"#10b981", "no2":"#f59e0b", "so2":"#6366f1", "o3":"#0ea5e9"}
    
    for p in selected_p:
        fig2.add_trace(go.Scatter(
            x=chart_df['date'], y=chart_df[p], name=PRETTY.get(p,p),
            line=dict(width=2, color=colors_map.get(p, "#888")),
            mode='lines'
        ))
    fig2.update_layout(
        template="plotly_white", 
        height=300, 
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>⚠️</span> Active Alerts</div>', unsafe_allow_html=True)
    
    alerts_html = ""
    if not pd.isna(current_aqi):
        if current_aqi > 300:
            alerts_html += f'<div class="alert-box" style="background-color:#fff1f2; border-color:#be123c; color:#881337"><span class="alert-icon">☢️</span><div class="alert-text">Severe Conditions!<br>Avoid outdoor exposure.</div></div>'
        elif current_aqi > 200:
            alerts_html += f'<div class="alert-box" style="background-color:#fff7ed; border-color:#c2410c; color:#7c2d12"><span class="alert-icon">😷</span><div class="alert-text">Poor Air Quality.<br>Sensitive groups wear masks.</div></div>'
        elif current_aqi <= 100:
            alerts_html += f'<div class="alert-box" style="background-color:#f0fdf4; border-color:#15803d; color:#14532d"><span class="alert-icon">✅</span><div class="alert-text">Air Quality is Good.<br>Enjoy the outdoors!</div></div>'
        else:
             alerts_html += f'<div class="alert-box" style="background-color:#fefce8; border-color:#ca8a04; color:#854d0e"><span class="alert-icon">ℹ️</span><div class="alert-text">Moderate Quality.<br>Exercise caution.</div></div>'
    else:
        alerts_html += "<div>No AQI data available for alerts.</div>"
        
    st.markdown(alerts_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- LAYOUT: POLLUTANT METRICS ----------------
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header"><span>🔬</span> Latest Readings</div>', unsafe_allow_html=True)

metrics_html = '<div class="metric-grid">'
for p in selected_p:
    val = latest_row.get(p)
    sub = sub_index(p, val)
    cat_l, cat_c = aqi_category_and_color(sub)
    
    metrics_html += f'<div class="metric-card">'
    metrics_html += f'<div class="metric-title">{PRETTY.get(p,p)}</div>'
    metrics_html += f'<div class="metric-value">{val:.1f}</div>'
    metrics_html += f'<div class="metric-sub" style="color:{cat_c}">Sub-Index: {sub:.0f} ({cat_l})</div>'
    metrics_html += '</div>'

metrics_html += '</div>'
st.markdown(metrics_html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RESTORED DATA SUMMARY ----------------
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header"><span>📊</span> Data Summary</div>', unsafe_allow_html=True)

sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
with sum_c1:
    st.metric("Total Records", len(fdf))
with sum_c2:
    st.metric("Average AQI", f"{fdf['AQI_calc'].mean():.1f}")
with sum_c3:
    st.metric("Max AQI", f"{fdf['AQI_calc'].max():.0f}")
with sum_c4:
    st.metric("Date Range", f"{len(pd.unique(fdf['date'].dt.date))} Days")

st.markdown("<div style='margin-top:10px; color:#64748b; font-size:0.85rem'><em>Note: AQI calculated using CPCB breakpoints. Missing values imputed using mean strategy.</em></div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PDF REPORT GENERATION ----------------
def create_pdf(dataframe, current_aqi_val):
    buffer = BytesIO()
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "Air Quality Report")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Generated on: {pd.Timestamp.now()}")
        c.drawString(50, height - 100, f"Location: {sel_loc}")
        c.drawString(50, height - 120, f"Average AQI (Selected Period): {dataframe['AQI_calc'].mean():.1f}")
        c.drawString(50, height - 140, f"Latest AQI: {current_aqi_val:.1f}")
        
        c.drawString(50, height - 180, "-------------------------------------------------------")
        c.drawString(50, height - 200, "Recent Data Logs (Last 10 entries):")
        
        y = height - 230
        c.setFont("Courier", 10)
        headers = "Date       | AQI | Status"
        c.drawString(50, y, headers)
        y -= 15
        
        recent = dataframe.sort_values('date', ascending=False).head(10)
        for idx, row in recent.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            aqi_val = row['AQI_calc']
            stat, _ = aqi_category_and_color(aqi_val)
            line = f"{date_str} | {aqi_val:3.0f} | {stat}"
            c.drawString(50, y, line)
            y -= 12
            
        c.save()
    except ImportError:
        return None
    return buffer.getvalue()

# Footer Row
c5, c6 = st.columns([1, 1])
with c5:
    if st.button("📄 Generate PDF Report"):
        pdf_data = create_pdf(fdf, current_aqi)
        if pdf_data:
            st.download_button("Download PDF", pdf_data, file_name="AirQualityReport.pdf", mime="application/pdf")
        else:
            st.error("ReportLab library not installed. Run `pip install reportlab`")

with c6:
    with st.expander("View Raw Data"):
        st.dataframe(fdf.sort_values('date', ascending=False), height=200)
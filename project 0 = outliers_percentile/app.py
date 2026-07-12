import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# 1. Page Configuration & Theme Setup
st.set_page_config(
    page_title="Outlier Detection Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI enhancements (Banners, Styling, Fonts)
st.markdown("""
    <style>
    /* Main Top Banner */
    .main-banner {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    /* Section Headers */
    .section-header {
        color: #4b6cb7;
        font-weight: 600;
        border-bottom: 2px solid #4b6cb7;
        padding-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Top Banner UI
st.markdown("""
    <div class="main-banner">
        <h1>📊 Outlier Detection & Percentile Analysis Studio</h1>
        <p>Analyze NYC Airbnb prices, filter anomalies, and clean your datasets instantly.</p>
    </div>
""", unsafe_allow_html=True)

# 2. Sidebar Navigation & "About" Section
st.sidebar.title("Configuration")
app_mode = st.sidebar.radio("Navigate", ["Analysis Dashboard", "About the App"])

if app_mode == "About the App":
    st.subheader("ℹ️ About the App")
    st.info("""
    **Outlier Detection Studio v1.0**  
    This application utilizes the **Percentile Method** to identify and remove extreme anomalies or data-entry errors in datasets. 
    
    *   **Core Logic:** It eliminates the bottom 1% and top 0.1% of pricing outliers to keep data distributions realistic.
    *   **Built With:** Python, Streamlit, Pandas, and Plotly.
    """)
    st.stop()

# 3. Load Dataset
@st.cache_data
def load_data():
    # Make sure AB_NYC_2019.csv is in the same folder
    return pd.read_csv("AB_NYC_2019.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ CSV file 'AB_NYC_2019.csv' not found. Please place it in the same directory.")
    st.stop()

# 4. Outlier Logic Controls
st.sidebar.subheader("Threshold Settings")
lower_pct = st.sidebar.slider("Lower Percentile Threshold", 0.0, 5.0, 1.0, step=0.1) / 100
upper_pct = st.sidebar.slider("Upper Percentile Threshold", 95.0, 99.99, 99.9, step=0.05) / 100

min_threshold, max_threshold = df.price.quantile([lower_pct, upper_pct])
df_cleaned = df[(df.price > min_threshold) & (df.price < max_threshold)]

# 5. UI Layout - Metrics Row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Original Rows", value=df.shape[0])
with col2:
    st.metric(label="Rows After Cleaning", value=df_cleaned.shape[0], delta=f"-{df.shape[0] - df_cleaned.shape[0]} Outliers")
with col3:
    # "Cylinder / Gauge" Visual for Data Retention Accuracy
    retention_rate = (df_cleaned.shape[0] / df.shape[0]) * 100
    st.metric(label="Data Retention Quality", value=f"{retention_rate:.2f}%")

# 6. Visualizations (Line Charts & Gauges)
st.markdown("<h3 class='section-header'>📈 Visualization Insights</h3>", unsafe_allow_html=True)
v_col1, v_col2 = st.columns([2, 1])

with v_col1:
    st.write("**Price Trend / Distribution Line Chart**")
    # Sample data for fast plotting performance
    sample_df = df_cleaned.sample(min(1000, df_cleaned.shape[0])).sort_values(by='price')
    fig_line = px.line(sample_df, y='price', title="Sampled Price Clean Distribution Trend", markers=True)
    fig_line.update_layout(template="plotly_dark") # Adapts to Dark Mode UI perfectly
    st.plotly_chart(fig_line, use_container_width=True)

with v_col2:
    st.write("**Data Integrity Cylinder Gauge**")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = retention_rate,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Accuracy Index (%)"},
        gauge = {
            'axis': {'range': [90, 100]},
            'bar': {'color': "#4b6cb7"},
            'steps': [
                {'range': [90, 95], 'color': "#ff4b4b"},
                {'range': [95, 100], 'color': "#00cc96"}
            ]
        }
    ))
    fig_gauge.update_layout(template="plotly_dark", height=280)
    st.plotly_chart(fig_gauge, use_container_width=True)

# 7. Data Sample Table View
st.markdown("<h3 class='section-header'>📋 Cleaned Dataset Preview</h3>", unsafe_allow_html=True)
st.dataframe(df_cleaned.sample(min(5, df_cleaned.shape[0])), use_container_width=True)
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. STREAMLIT PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="HR Analytics & Employee Retention Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar UI Selection for Theme & Sections
st.sidebar.title("🎨 Control Panel")
theme_choice = st.sidebar.radio("Choose App Theme:", ["Dark Mode UI", "Light Gradient UI"])
section_choice = st.sidebar.selectbox("Navigate Sections:", [
    "📌 Overview & Dataset", 
    "📈 Data Exploration & Visuals", 
    "🤖 About the Model Insights"
])

# Custom Injectable CSS Injection Framework
if theme_choice == "Dark Mode UI":
    css_theme = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .custom-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3, p, li { color: #f8fafc !important; }
    .footer {
        position: fixed;
        left: 0; bottom: 0; width: 100%;
        background-color: #0f172a;
        color: #94a3b8; text-align: center;
        padding: 10px; font-size: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 100;
    }
    </style>
    """
    plt_theme = 'dark_background'
    bar_colors = ['#6366f1', '#f43f5e'] # Indigo & Rose
else:
    css_theme = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
        color: #0f172a;
    }
    .custom-card {
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    h1, h2, h3, p, li { color: #0f172a !important; }
    .footer {
        position: fixed;
        left: 0; bottom: 0; width: 100%;
        background-color: #f8fafc;
        color: #64748b; text-align: center;
        padding: 10px; font-size: 14px;
        border-top: 1px solid rgba(0, 0, 0, 0.05);
        z-index: 100;
    }
    </style>
    """
    plt_theme = 'default'
    bar_colors = ['#3b82f6', '#ef4444'] # Blue & Red

st.markdown(css_theme, unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
@st.cache_data
def load_data():
    # Attempt loading directly from standard kaggle filename setup
    try:
        data = pd.read_csv("HR_comma_sep.csv")
    except FileNotFoundError:
        # Fallback empty simulator blueprint container for presentation scaffolding
        st.warning("Please place 'HR_comma_sep.csv' in the same directory as this file to load accurate structural calculations.")
        columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
                   'time_spend_company', 'Work_accident', 'left', 'promotion_last_5years', 'Department', 'salary']
        data = pd.DataFrame(columns=columns)
    return data

df = load_data()

# Global metrics used throughout sections
if not df.empty:
    left_count = df[df.left == 1].shape[0]
    retained_count = df[df.left == 0].shape[0]

# ==========================================
# SECTION 1: OVERVIEW & DATASET
# ==========================================
if section_choice == "📌 Overview & Dataset":
    st.title("📊 HR Analytics & Employee Retention Dashboard")
    
    st.markdown("""
    <div class="custom-card">
        <h3>About this Platform</h3>
        <p>This web workspace assesses patterns behind why employees leave operational spaces using interactive visualization pipelines derived from Kaggle metrics data.</p>
        <p><b>Data Source:</b> <a href="https://www.kaggle.com/giripujar/hr-analytics" target="_blank">Kaggle HR Analytics Dataset</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    if not df.empty:
        st.subheader("Data Blueprint Structural Glimpse")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Interactive UI Metric Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sample Size", f"{df.shape[0]}")
        with col2:
            st.metric("Total Workers Retained (Left = 0)", f"{retained_count}")
        with col3:
            st.metric("Total Workers Left (Left = 1)", f"{left_count}")

# ==========================================
# SECTION 2: EXPLORATION & VISUALIZATIONS
# ==========================================
elif section_choice == "📈 Data Exploration & Visuals":
    st.title("📈 Data Exploration Charts")
    
    if not df.empty:
        # Global Plot configurations setup
        plt.style.use(plt_theme)
        
        st.subheader("Retention Breakdown Profile matrix mapping")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="custom-card"><h4>Impact of Salary Tier on Employee Retention</h4>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            pd.crosstab(df.salary, df.left).plot(kind='bar', ax=ax, color=bar_colors)
            ax.set_ylabel("Employee Count")
            ax.set_xlabel("Salary Tier Level")
            plt.xticks(rotation=0)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="custom-card"><h4>Department-wise Retention Rates Metrics Matrix</h4>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            pd.crosstab(df.Department, df.left).plot(kind='bar', ax=ax, color=bar_colors)
            ax.set_ylabel("Employee Count")
            ax.set_xlabel("Department Segment")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        # Statistical aggregates display table matrix mapping frame
        st.subheader("Structural Mean Feature Distribution across Attrition Targets")
        st.dataframe(df.groupby('left').mean(numeric_only=True), use_container_width=True)
    else:
        st.info("No numerical attributes array found. Add standard csv source to build functional visual graphs charts updates context.")

# ==========================================
# SECTION 3: ABOUT THE MODEL INSIGHTS
# ==========================================
elif section_choice == "🤖 About the Model Insights":
    st.title("🤖 Operational Insights & Behavioral Models")
    
    st.markdown("""
    <div class="custom-card">
        <h3>💡 Critical Data Discovery Insights Matrix</h3>
        <p>From evaluating the mean values across feature arrays relative to structural separations (Left=1 vs Retained=0), we observe several structural patterns:</p>
        <ol>
            <li><b>Satisfaction Indices Level:</b> Employees leaving show distinct satisfaction drops (~0.44 average) vs those choosing retention tracks (~0.66).</li>
            <li><b>Workplace Overtime load (Average Monthly Hours):</b> Departing professionals register higher monthly shifts on average (207 hrs vs 199 hrs).</li>
            <li><b>Structural Promotions Track:</b> Staff rewarded with promotion upgrades within structural five-year tracks reflect a near-universal tendency to stay onboard.</li>
            <li><b>Financial Compensations (Salary Matrix):</b> Higher tier corporate tier salary structures insulate business environments against retention turnover losses.</li>
        </ol>
    </div>
    <div class="custom-card">
        <h3>🛠️ Next Steps For Machine Learning Deployment</h3>
        <p>To scale this logic into an interactive predictive classification instrument tool via this app screen setup configuration frame:</p>
        <ul>
            <li>Convert nominal factors (<i>Salary, Department</i>) via One-Hot Encoding schemas.</li>
            <li>Extract key target structural estimators matrix columns to build standard <code>LogisticRegression</code> or <code>RandomForestClassifier</code> matrices.</li>
            <li>Expose prediction user fields parameters right inside this custom layout context for high fidelity retention risk evaluation testing tools.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. FIXED FOOTER STRUCTURAL UI
# ==========================================
st.markdown("""
<div class="footer">
    📊 HR Analytics Dashboard Engine Framework Build v1.1 • Powered by Streamlit
</div>
""", unsafe_allow_html=True)
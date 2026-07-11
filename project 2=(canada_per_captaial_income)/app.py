
import streamlit as st
import pandas as pd
from sklearn import linear_model

# Set page configuration
st.set_page_config(
    page_title="Canada Per Capita Income Predictor",
    page_icon="🇨🇦",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Professional Landing Page Style Header ---
st.markdown(
    """
    <style>
    .header-style {
        background-color: #262730; /* Darker background for header */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .header-style h1 {
        color: #4CAF50; /* Green for main title */
        text-align: center;
        font-size: 3em;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px #000000;
    }
    .header-style h3 {
        color: #B0C4DE; /* Light blue for subtitle */
        text-align: center;
        font-size: 1.2em;
        font-style: italic;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2em;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-left: 5px solid #28a745;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
    <div class="header-style">
        <h1>📈 Canada Per Capita Income Predictor 🇨🇦</h1>
        <h3>Leveraging Linear Regression to Forecast Economic Trends</h3>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- House Image Banner (Placeholder) ---
st.image("https://upload.wikimedia.org/wikipedia/commons/d/d2/Parliament_Hill_%28Ottawa%29.jpg", 
         caption="Parliament Hill, Ottawa - A symbol of Canada's economic landscape", 
         use_column_width=True)
st.markdown("--- ")

# --- Load Data and Train Model ---
# Ensure 'canada_per_capita_income.csv' is in the same directory as app.py
try:
    df = pd.read_csv("canada_per_capita_income.csv")
except FileNotFoundError:
    st.error("Error: 'canada_per_capita_income.csv' not found. Please ensure the CSV file is in the same directory as `app.py`.")
    st.stop()

# Prepare data for model training
new_df = df[['year']]
price = df['per capita income (US$)']

# Create and train linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df, price)

# --- About the Model Section ---
st.header("🤖 About the Model")
st.write(
    "This interactive application utilizes a simple **Linear Regression** model to predict Canada's per capita income. "
    "The model learns the relationship between the 'year' and 'per capita income (US$)' from historical data (1970-2016) "
    "to make future forecasts."
)
st.subheader("Model Details:")
st.write(f"- **Coefficient (m):** `{reg.coef_[0]:,.2f}`")
st.write(f"- **Intercept (b):** `{reg.intercept_:,.2f}`")
st.info("A linear regression model finds the best-fitting straight line through the data points to predict a continuous outcome.")
st.markdown("--- ")

# --- Prediction Section ---
st.header("🔮 Predict Per Capita Income")
st.write("Enter a year below to get the predicted per capita income in US$.")

year_to_predict = st.number_input(
    "Enter a Year:",
    min_value=int(df['year'].min()),
    max_value=2050,
    value=2020,
    step=1,
    help="Select a year to see the predicted per capita income."
)

if st.button("Calculate Prediction"):

    # Predict for the given year
    prediction = float(reg.predict([[year_to_predict]])[0])

    # Success message
    st.success("Prediction calculated successfully!")

    # Display prediction in a styled card
    st.markdown(
        f"""
        <div style="
            background-color:#e8f5e9;
            border-left:8px solid #4CAF50;
            padding:20px;
            border-radius:10px;
            margin-top:15px;
        ">
            <h3 style="color:#2e7d32; margin-bottom:10px;">
                📊 Predicted Per Capita Income
            </h3>
            <p style="font-size:20px; margin:0;">
                <strong>Year:</strong> {year_to_predict}
            </p>
            <h2 style="color:#1b5e20; margin-top:10px;">
                💰 ${prediction:,.2f} USD
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.balloons()

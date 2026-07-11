import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import time

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

st.markdown("""
<style>
.stApp{background:#0E1117;color:white;}
h1,h2,h3,h4,h5,h6,p,label{color:white!important;}
section[data-testid="stSidebar"]{background:#161B22;}
.stButton>button{
background:linear-gradient(90deg,#00C6FF,#0072FF);
color:white;border:none;border-radius:10px;
height:50px;width:100%;font-size:18px;font-weight:bold;}
.prediction{
background:#1E293B;padding:25px;border-radius:16px;
text-align:center;font-size:30px;color:#00E5FF;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🏠 Project Information")
st.sidebar.markdown("""
**Algorithm:** Linear Regression

**Input:** House Area (sq.ft.)

**Output:** House Price

**Developer:** Akhilesh Yadav
""")

st.markdown("<h1 style='text-align:center;'>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict House Prices using Machine Learning & Linear Regression</p>", unsafe_allow_html=True)

st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa", use_container_width=True)

uploaded = st.file_uploader("Upload CSV (Optional)", type="csv")
df = pd.read_csv(uploaded) if uploaded else pd.read_csv("houseprice.csv")

with st.expander("📊 View Dataset"):
    st.dataframe(df)

st.subheader("Dataset Statistics")
st.dataframe(df.describe())

X = df.drop("price", axis=1)
y = df["price"]
model = LinearRegression()
model.fit(X, y)

col1,col2 = st.columns([2,1])

with col1:
    area = st.slider("House Area (Square Feet)",100,10000,3300,100)
    if st.button("Predict House Price"):
        with st.spinner("Predicting..."):
            time.sleep(1)
        pred = model.predict([[area]])[0]
        st.markdown(f"<div class='prediction'>Estimated Price<br><br>₹ {pred:,.2f}</div>", unsafe_allow_html=True)
        out = pd.DataFrame({"Area":[area],"Predicted Price":[pred]})
        st.download_button("Download Prediction CSV", out.to_csv(index=False), "prediction.csv","text/csv")

with col2:
    st.subheader("Model Details")
    st.metric("Coefficient", round(model.coef_[0],2))
    st.metric("Intercept", round(model.intercept_,2))

st.markdown("---")
st.header("🤖 About the Model")
st.write("""
This application uses **Linear Regression** to estimate the price of a house based on its area.
The model is trained on the uploaded/default dataset and predicts a continuous house price.
""")

st.markdown("---")
st.header("🏗️ Architecture Summary")
st.code("""House Dataset (CSV)
      │
      ▼
Data Preprocessing
      │
      ▼
Linear Regression
      │
      ▼
User Input
      │
      ▼
Prediction
      │
      ▼
Display Result""")

st.markdown("---")
st.markdown("<center>Made with ❤️ using Streamlit & Scikit-Learn<br>Developed by <b>Akhilesh Yadav</b></center>", unsafe_allow_html=True)

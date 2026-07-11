import streamlit as st
import numpy as np
from PIL import Image
import joblib

st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="👤",
    layout="centered"
)

model = joblib.load("Male_Female_model.pkl")

IMG_SIZE = 64

st.title("👤 Male vs Female Image Classifier")
st.write("Upload an image to predict whether it is Male or Female.")

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    resized = image.resize((IMG_SIZE, IMG_SIZE))
    resized = np.array(resized)
    resized = resized.flatten()

    prediction = model.predict([resized])[0]
    probability = model.predict_proba([resized])[0]

    if prediction == 0:
        st.success("👨 Prediction: MALE")
    else:
        st.success("👩 Prediction: FEMALE")

    st.subheader("Prediction Confidence")
    st.write(f"👨 Male Probability: **{probability[0] * 100:.2f}%**")
    st.write(f"👩 Female Probability: **{probability[1] * 100:.2f}%**")

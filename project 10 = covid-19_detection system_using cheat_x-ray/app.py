import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# --- Custom CSS for a better look --- #
st.markdown(
    """
    <style>
    .main-header {color: #FF4B4B; text-align: center;}
    .sidebar-content {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stAlert {background-color: #e6ffe6; color: #4CAF50;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Streamlit App Layout --- #

# Main Title with custom styling
st.markdown('<h1 class="main-header">COVID-19 Chest X-Ray Classifier</h1>', unsafe_allow_html=True)

# Sidebar for information or additional features
st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
st.sidebar.header("About This App")
st.sidebar.write(
    "This application uses a Convolutional Neural Network (CNN) "
    "to classify chest X-ray images as either 'COVID' or 'NORMAL'. "
    "The model was trained on a dataset of chest X-ray images."
)
st.sidebar.info("Disclaimer: This app is for educational purposes only and should not be used for medical diagnosis.")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# --- Model Loading --- #

# Use st.cache_resource to load the model only once. This is crucial for performance.
@st.cache_resource
def load_my_model():
    # Make sure 'my_model.keras' is in the same directory as app.py
    # when you deploy your app, or provide a full path if stored elsewhere (e.g., cloud storage).
    try:
        model = load_model('my_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'my_model.keras' is in the correct path.")
        return None

model = load_my_model()

if model is None:
    st.stop() # Stop the app if the model couldn't be loaded

# --- Main Content Area --- #

st.write("Upload a chest X-ray image (JPG, JPEG, PNG) to get a COVID-19 prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Use a container for the image and prediction results
    with st.container(border=True):
        # Display the uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image.', use_column_width=True)

        with col2:
            st.subheader("Prediction Results")
            st.write("Classifying...")

            # Preprocess the image
            img_resized = img.resize((299, 299)) # Resize PIL Image to target size
            img_array = image.img_to_array(img_resized) / 255.0 # Convert to numpy array and normalize
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

            # Make prediction
            result = model.predict(img_array)

            # Assuming your model output has two classes (COVID and Normal)
            threshold = 0.5
            prediction_label = "COVID" if result[0][0] > threshold else "NORMAL"

            if prediction_label == "COVID":
                st.error(f"Prediction: **{prediction_label}**")
            else:
                st.success(f"Prediction: **{prediction_label}**")

            # Optional: Display raw prediction value
            st.info(f"Raw prediction score (probability for COVID): {result[0][0]:.4f}")

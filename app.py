import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Eye Gender Detection",
    page_icon="👁️",
    layout="centered"
)

# 1. Load the trained Keras model
@st.cache_resource
def load_model():
    # Make sure 'my_model.keras' is in the same directory as app.py
    return tf.keras.models.load_model('my_model.keras')

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please ensure 'my_model.keras' is placed in the same directory as this script.")
    model_loaded = False

# 2. User Interface
st.title("👁️ Male vs Female Eye Classifier")
st.write("Upload a photograph of a human eye to detect if it's classified as male or female.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("🔄 Processing and classifying...")
    
    # 3. Preprocessing to match your CNN input shape
    # Adjust target_size if your model uses a different resolution (e.g., 150x150)
    target_size = (150, 150) 
    
    # Convert to RGB if grayscale, resize, and convert to numpy array
    img_resized = image.convert("RGB").resize(target_size)
    img_array = np.array(img_resized)
    
    # Normalize pixel values if your ImageDataGenerator used rescaling (e.g., rescale=1./255)
    img_array = img_array / 255.0
    
    # Expand dimensions to create batch size of 1 -> (1, 150, 150, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 4. Model Prediction
    prediction = model.predict(img_batch)
    
    # Depending on how your final layer is configured (Binary Sigmoid output):
    # Usually, ImageDataGenerator assigns labels alphabetically: Female = 0, Male = 1 (or vice versa)
    # Check your class_indices from training to verify. Assuming 0 = Female, 1 = Male:
    score = prediction[0][0]
    
    st.markdown("---")
    st.subheader("Results:")
    
    if score > 0.5:
        confidence = score * 100
        st.success(f"Prediction: **Male Eye** (Confidence: {confidence:.2f}%)")
    else:
        confidence = (1 - score) * 100
        st.success(f"Prediction: **Female Eye** (Confidence: {confidence:.2f}%)")
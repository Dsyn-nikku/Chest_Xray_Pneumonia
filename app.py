
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩻",
    layout="wide"
)

# Load model
MODEL_PATH = Path("models/chest_xray_model.keras")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("🩻 Chest X-ray Pneumonia Detection")
st.write("Upload a chest X-ray image for an educational AI classification.")

st.warning(
    "Educational prototype only. This application is not a medical "
    "diagnostic tool. Do not use predictions for medical decisions."
)

# Check model
if not MODEL_PATH.exists():
    st.error("Model not found. Please train your model first using train.py.")
    st.stop()

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded X-ray")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        if st.button("Predict Pneumonia", type="primary"):

            # Preprocessing
            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            # Model prediction
            prediction = float(
                model.predict(image_array, verbose=0)[0][0]
            )

            if prediction >= 0.5:
                result = "PNEUMONIA"
                score = prediction
            else:
                result = "NORMAL"
                score = 1 - prediction

            st.success(f"Predicted class: {result}")

            st.metric(
                "Model score for predicted class",
                f"{score * 100:.2f}%"
            )

            st.caption(
                "This score is not a calibrated probability of disease "
                "and is not medical certainty."
            )

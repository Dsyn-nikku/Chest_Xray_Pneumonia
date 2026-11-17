
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩻",
    layout="wide"
)

# Model path
MODEL_PATH = Path(__file__).parent / "models" / "chest_xray_model.h5"


# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


# Application title
st.title("🩻 Chest X-ray Pneumonia Detection")

st.write(
    "Upload a chest X-ray image for an educational AI classification."
)

st.warning(
    "Educational prototype only. This application is not a medical "
    "diagnostic tool. Do not use predictions for medical decisions."
)


# Check model availability
if not MODEL_PATH.exists():
    st.error("Model not found! Please check the models folder.")
    st.stop()


# Load model
try:
    model = load_model()
    st.success("AI Model loaded successfully!")

except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# Upload image
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:
        # Read image safely
        image_bytes = uploaded_file.getvalue()

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        col1, col2 = st.columns(2)

        # Display uploaded image
        with col1:
            st.subheader("Uploaded X-ray")

            st.image(
                  image_bytes,
                  caption="Uploaded Chest X-ray",
                  width=400,
                 output_format="JPEG"
               )
        # Prediction section
        with col2:
            st.subheader("Prediction")

            if st.button(
                "Predict Pneumonia",
                type="primary"
            ):

                # Preprocessing
                processed_image = image.resize((224, 224))

                image_array = np.array(
                    processed_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Model prediction
                prediction = float(
                    model.predict(
                        image_array,
                        verbose=0
                    )[0][0]
                )

                # Classification
                if prediction >= 0.5:
                    result = "PNEUMONIA"
                    score = prediction
                else:
                    result = "NORMAL"
                    score = 1 - prediction

                st.success(
                    f"Predicted class: {result}"
                )

                st.metric(
                    "Model score for predicted class",
                    f"{score * 100:.2f}%"
                )

                st.caption(
                    "This score is not a calibrated probability "
                    "of disease and is not medical certainty."
                )

    except Exception as e:
        st.error("Unable to display or process this image.")
        st.exception(e)

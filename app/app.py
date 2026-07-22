import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ------------------------------------
# PAGE CONFIG
# ------------------------------------

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Classification")

st.write(
    "Upload an MRI image and predict the tumor type using a MobileNetV2 Deep Learning model."
)

# ------------------------------------
# SIDEBAR
# ------------------------------------

st.sidebar.title("🧠 Brain Tumor Classification")

st.sidebar.markdown("---")

st.sidebar.header("📌 Model Information")

st.sidebar.write("**Model:** MobileNetV2")

st.sidebar.write("**Framework:** TensorFlow / Keras")

st.sidebar.write("**Number of Classes:** 4")

st.sidebar.markdown("### Classes")

st.sidebar.write("🟢 Glioma")

st.sidebar.write("🟠 Meningioma")

st.sidebar.write("🔵 No Tumor")

st.sidebar.write("🟣 Pituitary")

st.sidebar.markdown("---")

st.sidebar.success("✅ Test Accuracy : 89.88%")

# ------------------------------------
# LOAD MODEL
# ------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "brain_tumor_mobilenetv2.keras"
)

@st.cache_resource
def load_brain_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_brain_model()

# ------------------------------------
# CLASS NAMES
# ------------------------------------

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# ------------------------------------
# IMAGE UPLOAD
# ------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    # ------------------------------------
    # PREPROCESS IMAGE
    # ------------------------------------

    img = image.resize((224,224))

    img_array = np.array(img).astype(np.float32)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # ------------------------------------
    # PREDICT BUTTON
    # ------------------------------------

    if st.button("🔍 Predict"):

        prediction = model.predict(img_array, verbose=0)

        predicted_class = np.argmax(prediction[0])

        confidence = prediction[0][predicted_class]

        st.success("✅ Prediction Completed")

        st.markdown("---")

        st.subheader("🧠 Prediction Result")

        st.metric(
            label="Tumor Type",
            value=class_names[predicted_class]
        )

        st.metric(
            label="Confidence",
            value=f"{confidence*100:.2f}%"
        )

        st.markdown("---")

        st.subheader("📊 Probability Distribution")

        for i, cls in enumerate(class_names):

            probability = float(prediction[0][i])

            st.write(f"**{cls}**")

            st.progress(probability)

            st.write(f"{probability*100:.2f}%")

            st.write("")
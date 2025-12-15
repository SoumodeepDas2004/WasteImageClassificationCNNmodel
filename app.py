# Copyright (c) 2025 Soumodeep Das
# Licensed under the Apache License 2.0

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- CONFIG ----------------
MODEL_PATH = "PlasticWasteClassifier_v1.h5"
IMG_SIZE = 224
CLASS_NAMES = ["Organic", "Recyclable"]
# ----------------------------------------

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

st.set_page_config(
    page_title="Plastic Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(-45deg, #0d5e36, #203a43, #6315bd);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    color: #00ffcc;
}

/* Card */
.card {
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.3);
    margin-top: 20px;
}

/* Upload box */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 15px;
}

/* Prediction text */
.prediction {
    font-size: 1.6rem;
    font-weight: bold;
    color: #00ff99;
    text-align: center;
}

/* Confidence */
.confidence {
    font-size: 1.2rem;
    color: #ffdd57;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.markdown('<div class="title">♻️ Plastic Waste Classification</div>', unsafe_allow_html=True)

st.markdown("""
<div class="card">
    Upload an image to classify <b>Organic</b> or <b>Recyclable</b> waste using a CNN model.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction">🧠 {CLASS_NAMES[class_index]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

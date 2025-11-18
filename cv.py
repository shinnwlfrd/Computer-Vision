import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import joblib

cnn = load_model("cnn_feature_extractor.h5")
knn = joblib.load("knn_classifier.pkl")

IMG_SIZE = 128

st.title("Fresh vs Non-Fresh Fish Classifier")

uploaded = st.file_uploader("Upload gambar ikan", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    st.image(img, caption="Gambar diupload", width=300)

    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    feature = cnn.predict(arr)
    pred = knn.predict(feature)

    result = "fresh" if pred[0] == 1 else "nonfresh"

    st.subheader("Hasil Prediksi: " + result)

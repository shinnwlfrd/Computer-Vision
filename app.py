import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import cv2
import os

# ============================
# STYLE (mirip aplikasi jurnal)
# ============================
st.set_page_config(
    page_title="Deteksi Kesegaran Ikan - Hybrid CNN + KNN",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #1f4e78;
        text-align: center;
        padding-bottom: 20px;
    }
    .sub-box {
        background: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #1f4e78;
    }
    .result-box {
        background: #e8f4ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #005bbb;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# LOAD MODEL (cached)
# ============================
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model("model/model_cnn.h5")
    knn_model = joblib.load("model/model_knn.pkl")
    return cnn_model, knn_model

cnn, knn = load_models()

# ============================
# IMAGE PREPROCESSING
# ============================
IMG_SIZE = 128

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = img.convert("L")        # grayscale
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=[0, -1])
    return img

def hybrid_predict(img):
    feature = cnn.predict(img)
    pred = knn.predict(feature)[0]
    return "fresh" if pred == 1 else "nonfresh"

# ============================
# UI
# ============================
st.markdown("<div class='main-title'>Aplikasi Deteksi Kesegaran Ikan (Hybrid CNN + KNN)</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

# ============================
# LEFT SIDE
# ============================
with col1:
    st.markdown("<div class='sub-box'>📤 <b>Upload Gambar Ikan</b></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Pilih gambar ikan (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=350, caption="Gambar yang diupload", use_column_width=False)

        st.markdown("<div class='sub-box'>🔍 <b>Ekstraksi Ciri Citra</b></div>", unsafe_allow_html=True)

        processed_img = preprocess_image(img)
        st.success("Ekstraksi selesai! Citra siap diklasifikasi.")

        st.markdown("<div class='sub-box'>🤖 <b>Klasifikasi Hybrid CNN + KNN</b></div>", unsafe_allow_html=True)

        if st.button("Prediksi"):
            result = hybrid_predict(processed_img)

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.write("### Hasil Prediksi:")
            st.write(f"**Ikan terdeteksi sebagai:** `{result.upper()}`")
            st.markdown("</div>", unsafe_allow_html=True)


# ============================
# RIGHT SIDE – INFO PANEL
# ============================
with col2:
    st.markdown("<div class='sub-box'>ℹ️ <b>Informasi Aplikasi</b></div>", unsafe_allow_html=True)
    st.write("""
Aplikasi ini dibuat untuk mendeteksi kesegaran ikan menggunakan **Hybrid CNN + KNN**.
Model CNN menghasilkan fitur citra, KNN melakukan klasifikasi akhir.

**Fitur utama:**
- Upload gambar ikan
- Ekstraksi ciri otomatis
- Prediksi segar atau tidak segar
- Antarmuka mirip desain GUI penelitian
    """)

    st.markdown("<div class='sub-box'>📘 <b>Detail Model</b></div>", unsafe_allow_html=True)
    st.write("""
- CNN digunakan sebagai feature extractor  
- KNN digunakan sebagai classifier  
- Input citra grayscale 128x128  
- Model dimuat sekali dengan caching agar cepat  
""")

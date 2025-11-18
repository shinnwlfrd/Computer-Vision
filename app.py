import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image

# ============================
# PAGE SETTINGS
# ============================
st.set_page_config(
    page_title="Deteksi Kesegaran Ikan",
    layout="wide"
)

# ============================
# CUSTOM DARK THEME + REMOVE WHITE BOX
# ============================
st.markdown("""
<style>

/* REMOVE STREAMLIT WHITE BLOCKS */
.css-1d391kg, .css-1iyw2u1, .css-12oz5g7, .css-18e3th9 {
    background-color: transparent !important;
    box-shadow: none !important;
}

/* MAIN BACKGROUND */
main {
    background-color: #0f172a;
}

/* TEXT COLOR */
body, p, div, span {
    color: #e2e8f0 !important;
}

/* HEADER TITLE */
.main-title {
    font-size: 34px;
    font-weight: bold;
    text-align: center;
    color: #60a5fa;
    padding: 10px 0 30px 0;
}

/* CUSTOM CARD */
.card {
    background: rgba(255,255,255,0.05);
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* RESULT BOX */
.result-box {
    background: rgba(59,130,246,0.15);
    padding: 18px;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
}

/* BUTTON STYLE */
.stButton > button {
    width: 100%;
    background: #2563eb !important;
    color: white !important;
    padding: 10px;
    border-radius: 10px;
    font-size: 16px;
    border: none;
}
.stButton > button:hover {
    background: #1e40af !important;
}

/* SIDEBAR DARK */
[data-testid="stSidebar"] {
    background-color: #1e293b !important;
}

</style>
""", unsafe_allow_html=True)

# ============================
# LOAD MODELS
# ============================
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model("cnn_feature_extractor.h5")
    knn_model = joblib.load("knn_classifier.pkl")
    return cnn_model, knn_model

cnn, knn = load_models()

IMG_SIZE = 128

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE)).convert("L")
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=[0, -1])
    return arr

def hybrid_predict(arr):
    feature = cnn.predict(arr)
    label = knn.predict(feature)[0]
    return "fresh" if label == 1 else "nonfresh"

# ============================
# UI LAYOUT
# ============================
st.markdown("<div class='main-title'>Deteksi Kesegaran Ikan (Hybrid CNN + KNN)</div>", unsafe_allow_html=True)

left, right = st.columns([1.3, 1])

# LEFT SIDE
with left:
    st.markdown("<div class='card'><b>📤 Upload Gambar Ikan</b></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Pilih gambar ikan", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar diupload", width=300)

        st.markdown("<div class='card'><b>🔍 Ekstraksi Ciri Citra</b></div>", unsafe_allow_html=True)
        arr = preprocess_image(img)
        st.success("Ekstraksi selesai.")

        st.markdown("<div class='card'><b>🤖 Klasifikasi</b></div>", unsafe_allow_html=True)
        if st.button("Prediksi"):
            result = hybrid_predict(arr)

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.write("### Hasil Prediksi:")
            st.write(f"**Ikan terdeteksi sebagai:** `{result.upper()}`")
            st.markdown("</div>", unsafe_allow_html=True)

# RIGHT SIDE
with right:
    st.markdown("<div class='card'><b>ℹ️ Informasi Aplikasi</b></div>", unsafe_allow_html=True)
    st.write("""
Aplikasi ini mendeteksi kesegaran ikan menggunakan Hybrid CNN + KNN:
- CNN sebagai extractor fitur
- KNN sebagai classifier
- Input citra grayscale 128×128  
- Model dimuat sekali (cached)
""")

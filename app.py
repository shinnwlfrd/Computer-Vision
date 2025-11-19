import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image

# ============================
# PAGE SETTINGS
# ============================
st.set_page_config(
    page_title="🐟 Deteksi Kesegaran Ikan",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# ENHANCED RESPONSIVE THEME
# ============================
st.markdown("""
<style>
/* IMPORT GOOGLE FONTS */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* RESET */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Inter', sans-serif;
}

/* BACKGROUND & BASE COLORS */
@media (prefers-color-scheme: dark) {
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        color: #e2e8f0 !important;
    }
    
    .gradient-header {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .info-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
    }
    
    .result-fresh {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%) !important;
        border-left: 5px solid #10b981 !important;
    }
    
    .result-nonfresh {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%) !important;
        border-left: 5px solid #ef4444 !important;
    }
    
    .stat-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] {
        background: rgba(30, 41, 59, 0.95) !important;
        backdrop-filter: blur(10px);
    }
    
    .stMarkdown, .stText, p, div, span, li, h1, h2, h3, h4, h5, h6 {
        color: #e2e8f0 !important;
    }
}

@media (prefers-color-scheme: light) {
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%) !important;
        color: #1e293b !important;
    }
    
    .gradient-header {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .info-card {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.05) 0%, rgba(124, 58, 237, 0.05) 100%) !important;
        border: 1px solid rgba(37, 99, 235, 0.15) !important;
    }
    
    .result-fresh {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.15) 100%) !important;
        border-left: 5px solid #10b981 !important;
    }
    
    .result-nonfresh {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%) !important;
        border-left: 5px solid #ef4444 !important;
    }
    
    .stat-box {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(0, 0, 0, 0.08);
    }
    
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
    }
    
    .stMarkdown, .stText, p, div, span, li, h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
    }
}

/* GLOBAL STYLES */
.main-header {
    text-align: center;
    padding: 30px 0 40px 0;
    animation: fadeInDown 0.8s ease-out;
}

.main-title {
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: -1px;
}

.subtitle {
    font-size: 18px;
    opacity: 0.8;
    font-weight: 400;
}

.card {
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease-out;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15) !important;
}

.card-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.icon {
    font-size: 24px;
}

.info-card {
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 20px;
}

.result-fresh, .result-nonfresh {
    padding: 24px;
    border-radius: 16px;
    margin-top: 20px;
    animation: scaleIn 0.5s ease-out;
}

.result-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 12px;
}

.result-label {
    font-size: 32px;
    font-weight: 700;
    margin-top: 8px;
    letter-spacing: 1px;
}

.stat-box {
    padding: 16px;
    border-radius: 12px;
    margin: 10px 0;
    text-align: center;
}

.stat-value {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 4px;
}

.stat-label {
    font-size: 14px;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.feature-list {
    list-style: none;
    padding: 0;
}

.feature-list li {
    padding: 12px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    align-items: center;
    gap: 12px;
}

.feature-list li:last-child {
    border-bottom: none;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: white !important;
    padding: 16px 24px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.6) !important;
}

.stFileUploader {
    border: 2px dashed rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 20px;
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: rgba(59, 130, 246, 0.6);
}

/* ANIMATIONS */
@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes scaleIn {
    from {
        opacity: 0;
        transform: scale(0.9);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

/* HIDE STREAMLIT BRANDING */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* REMOVE DEFAULT STREAMLIT STYLING */
section, .css-1d391kg, .css-1iyw2u1, .css-12oz5g7, .css-18e3th9 {
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# ============================
# LOAD MODELS
# ============================
@st.cache_resource
def load_models():
    try:
        cnn_model = tf.keras.models.load_model("cnn_feature_extractor.h5")
        knn_model = joblib.load("knn_classifier.pkl")
        return cnn_model, knn_model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

with st.spinner("🔄 Memuat model AI..."):
    cnn, knn = load_models()
IMG_SIZE = 128

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE)).convert("L")
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=[0, -1])
    return arr

def hybrid_predict(arr):
    feature = cnn.predict(arr, verbose=0)
    label = knn.predict(feature)[0]
    return "fresh" if label == 1 else "nonfresh"

# ============================
# HEADER
# ============================
st.markdown("""
<div class='main-header'>
    <h1 class='main-title gradient-header'>🐟 Deteksi Kesegaran Ikan</h1>
    <p class='subtitle'>Sistem Klasifikasi Hybrid CNN + KNN Berbasis AI</p>
</div>
""", unsafe_allow_html=True)

# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.markdown("### ⚙️ Informasi Model")
    st.markdown("""
    <div class='stat-box'>
        <div class='stat-value'>CNN</div>
        <div class='stat-label'>Ekstraksi Fitur</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='stat-box'>
        <div class='stat-value'>KNN</div>
        <div class='stat-label'>Klasifikasi</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='stat-box'>
        <div class='stat-value'>128×128</div>
        <div class='stat-label'>Ukuran Gambar</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Cara Kerja")
    st.markdown("""
    <ul class='feature-list'>
        <li>📤 Upload gambar ikan</li>
        <li>🔄 Preprocessing ke grayscale</li>
        <li>🧠 Ekstraksi fitur dengan CNN</li>
        <li>🎯 Klasifikasi dengan KNN</li>
        <li>✅ Dapatkan hasil instan</li>
    </ul>
    """, unsafe_allow_html=True)

# ============================
# MAIN LAYOUT
# ============================
left, right = st.columns([1.4, 1], gap="large")

with left:
    # Upload Section
    st.markdown("""
    <div class='card'>
        <div class='card-title'>
            <span class='icon'>📤</span>
            <span>Upload Gambar Ikan</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Pilih file gambar (JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)

        # Feature Extraction
        st.markdown("""
        <div class='card'>
            <div class='card-title'>
                <span class='icon'>🔍</span>
                <span>Ekstraksi Fitur</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔄 Memproses gambar..."):
            arr = preprocess_image(img)
            st.success("✅ Ekstraksi fitur berhasil diselesaikan!")

        # Classification
        st.markdown("""
        <div class='card'>
            <div class='card-title'>
                <span class='icon'>🤖</span>
                <span>Klasifikasi AI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Analisis Kesegaran"):
            with st.spinner("🧠 AI sedang menganalisis..."):
                result = hybrid_predict(arr)
                
                if result == "fresh":
                    st.markdown(f"""
                    <div class='result-fresh'>
                        <div class='result-title'>✅ Hasil Klasifikasi</div>
                        <p>Ikan telah diklasifikasikan sebagai:</p>
                        <div class='result-label'>🟢 SEGAR</div>
                        <p style='margin-top: 12px; opacity: 0.9;'>
                            Ikan tampak dalam kondisi baik dan aman untuk dikonsumsi.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                    <div class='result-nonfresh'>
                        <div class='result-title'>⚠️ Hasil Klasifikasi</div>
                        <p>Ikan telah diklasifikasikan sebagai:</p>
                        <div class='result-label'>🔴 TIDAK SEGAR</div>
                        <p style='margin-top: 12px; opacity: 0.9;'>
                            Ikan mungkin tidak cocok untuk dikonsumsi. Harap verifikasi sebelum digunakan.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

with right:
    # Information Card
    st.markdown("""
    <div class='info-card'>
        <div class='card-title'>
            <span class='icon'>ℹ️</span>
            <span>Tentang Aplikasi Ini</span>
        </div>
        <p style='line-height: 1.8; margin-top: 12px;'>
            Sistem cerdas ini menggabungkan kekuatan <strong>Convolutional Neural Networks (CNN)</strong> 
            untuk ekstraksi fitur dan <strong>K-Nearest Neighbors (KNN)</strong> untuk klasifikasi 
            guna mendeteksi kesegaran ikan secara akurat.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical Specifications
    st.markdown("""
    <div class='card'>
        <div class='card-title'>
            <span class='icon'>⚡</span>
            <span>Spesifikasi Teknis</span>
        </div>
        <ul class='feature-list'>
            <li>🧠 <strong>Deep Learning:</strong> Arsitektur CNN</li>
            <li>🎯 <strong>Klasifikasi:</strong> K-Nearest Neighbors</li>
            <li>🖼️ <strong>Format Input:</strong> Gambar grayscale</li>
            <li>📐 <strong>Resolusi:</strong> 128×128 piksel</li>
            <li>⚡ <strong>Performa:</strong> Model ter-cache untuk kecepatan</li>
            <li>🔒 <strong>Keandalan:</strong> Pendekatan hybrid</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Tips Section
    st.markdown("""
    <div class='info-card'>
        <div class='card-title'>
            <span class='icon'>💡</span>
            <span>Tips untuk Hasil Terbaik</span>
        </div>
        <ul class='feature-list'>
            <li>📸 Gunakan foto yang jelas dan terang</li>
            <li>🎯 Fokus pada badan ikan</li>
            <li>🔍 Hindari gambar yang buram</li>
            <li>📏 Pastikan framing yang tepat</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

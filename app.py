import os
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import base64

st.set_page_config(page_title="UTK Face Age & Gender Predictor", page_icon="👤", layout="wide")

# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "age_gender_model.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# ====================== STYLE ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500&display=swap');

#MainMenu, footer, header {visibility: hidden;}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #0F1626 0%, #0A0F1C 45%, #070B14 100%);
    color: #E5E7EB;
}

/* Headline */
.scan-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    letter-spacing: -0.02em;
    color: #F1F5F9;
    margin-bottom: 0.1rem;
}
.scan-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    color: #2DD4BF;
    text-transform: uppercase;
    margin-bottom: 1.6rem;
}

/* Panel */
.panel {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 1.3rem 1.4rem;
}
.panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    color: #64748B;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.panel-label::before {
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #2DD4BF;
    box-shadow: 0 0 6px #2DD4BF;
}

/* Sliders */
div[data-testid="stSlider"] label p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background-color: #2DD4BF !important;
    box-shadow: 0 0 8px rgba(45, 212, 191, 0.6) !important;
}
div[data-testid="stTickBar"] { display: none; }

/* File uploader */
[data-testid="stFileUploader"] section {
    background: #0D1420;
    border: 1.5px dashed #2A3549;
    border-radius: 10px;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #2DD4BF;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-family: 'JetBrains Mono', monospace;
    color: #64748B;
}

/* Viewfinder frame around cropped face */
.viewfinder {
    position: relative;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #1F2937;
}
.viewfinder::before, .viewfinder::after {
    content: "";
    position: absolute;
    width: 22px;
    height: 22px;
    z-index: 2;
}
.viewfinder .corner-tl, .viewfinder .corner-br {
    content: "";
    position: absolute;
    width: 22px;
    height: 22px;
    z-index: 2;
}
.viewfinder::before {
    top: 8px; left: 8px;
    border-top: 3px solid #2DD4BF;
    border-left: 3px solid #2DD4BF;
}
.viewfinder::after {
    bottom: 8px; right: 8px;
    border-bottom: 3px solid #2DD4BF;
    border-right: 3px solid #2DD4BF;
}
.corner-tl {
    top: 8px; right: 8px;
    border-top: 3px solid #2DD4BF;
    border-right: 3px solid #2DD4BF;
}
.corner-br {
    bottom: 8px; left: 8px;
    border-bottom: 3px solid #2DD4BF;
    border-left: 3px solid #2DD4BF;
}
.viewfinder img {
    display: block;
    width: 100%;
}
.viewfinder-caption {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: #64748B;
    text-transform: uppercase;
    margin-top: 0.6rem;
    text-align: center;
}

/* Readout rows */
.readout-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.85rem 0;
    border-bottom: 1px solid #1F2937;
}
.readout-row:last-child { border-bottom: none; }
.readout-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    color: #64748B;
    text-transform: uppercase;
}
.readout-val {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    color: #F1F5F9;
}
.readout-val.accent { color: #2DD4BF; }
.readout-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #475569;
}

.confidence-track {
    width: 100%;
    height: 5px;
    background: #1F2937;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 0.4rem;
}
.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #2DD4BF, #5EEAD4);
}

.raw-line {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #3F4C5F;
    margin-top: 1.1rem;
    padding-top: 0.8rem;
    border-top: 1px dashed #1F2937;
}

.empty-state {
    font-family: 'JetBrains Mono', monospace;
    color: #475569;
    font-size: 0.85rem;
    text-align: center;
    padding: 3rem 1rem;
    border: 1px dashed #1F2937;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model Not Found\nPath: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ====================== FUNCTIONS ======================
def crop_face(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if not os.path.exists(CASCADE_PATH):
        return img_rgb

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
        margin = int(0.3 * w)
        return img_rgb[max(0, y-margin):y+h+margin, max(0, x-margin):x+w+margin]
    return img_rgb

def preprocess(cropped):
    if cropped is None:
        return None
    resized = cv2.resize(cropped, (224, 224))
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR).astype(np.float32)
    bgr[..., 0] -= 103.939
    bgr[..., 1] -= 116.779
    bgr[..., 2] -= 123.68
    return np.expand_dims(bgr, 0)

# ====================== HEADER ======================
st.markdown('<div class="scan-title">Face Age &amp; Gender Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="scan-subtitle">// Upload a photo to run a scan</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 3], gap="large")

# ====================== CONTROLS PANEL ======================
with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Scan Controls</div>', unsafe_allow_html=True)
    max_age = st.slider("Max Age", 1, 116, 116)
    bias = st.slider("Fine Tune", -15, 15, 0, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<p style="font-family:JetBrains Mono, monospace; font-size:0.7rem; color:#3F4C5F; margin-top:1rem;">'
        'DEV // Satyam Kumar</p>',
        unsafe_allow_html=True
    )

# ====================== MAIN AREA ======================
with col2:
    uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        uploaded_file.seek(0)
        face = crop_face(uploaded_file)

        if face is not None:
            left, right = st.columns([1, 1.1], gap="large")

            with left:
                st.markdown('<div class="panel-label">Detected Face</div>', unsafe_allow_html=True)
                buf_ok, buf = cv2.imencode(".png", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                b64 = base64.b64encode(buf).decode()
                st.markdown(f"""
                    <div class="viewfinder">
                        <div class="corner-tl"></div>
                        <div class="corner-br"></div>
                        <img src="data:image/png;base64,{b64}"/>
                    </div>
                    <div class="viewfinder-caption">Auto-cropped &middot; face-locked</div>
                """, unsafe_allow_html=True)

            with right:
                st.markdown('<div class="panel-label">Scan Result</div>', unsafe_allow_html=True)
                with st.spinner("Analyzing..."):
                    tensor = preprocess(face)
                    preds = model.predict(tensor, verbose=0)
                    if isinstance(preds, list):
                        age_raw = float(preds[0][0][0])
                        gender_raw = float(preds[1][0][0])
                    else:
                        flat = np.array(preds).flatten()
                        age_raw = float(flat[0])
                        gender_raw = float(flat[1]) if len(flat) > 1 else 0.5

                    predicted_age = int(round(age_raw * 19.924 + 33.246)) + bias
                    predicted_age = max(1, min(predicted_age, max_age))
                    gender = "Female" if gender_raw >= 0.5 else "Male"
                    gender_conf = max(gender_raw, 1.0 - gender_raw)

                st.markdown(f"""
                    <div class="panel">
                        <div class="readout-row">
                            <span class="readout-key">Gender</span>
                            <span class="readout-val accent">{gender}</span>
                        </div>
                        <div class="readout-row" style="flex-direction:column; align-items:stretch;">
                            <div style="display:flex; justify-content:space-between;">
                                <span class="readout-key">Confidence</span>
                                <span class="readout-sub">{gender_conf*100:.1f}%</span>
                            </div>
                            <div class="confidence-track">
                                <div class="confidence-fill" style="width:{gender_conf*100:.1f}%;"></div>
                            </div>
                        </div>
                        <div class="readout-row">
                            <span class="readout-key">Estimated Age</span>
                            <span class="readout-val">{predicted_age} <span class="readout-sub">yrs</span></span>
                        </div>
                        <div class="raw-line">raw_output: {age_raw:.4f}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state">Could not read this image file.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">Awaiting upload &mdash; drop a photo above to begin a scan.</div>', unsafe_allow_html=True)

st.markdown(
    '<p style="font-family:JetBrains Mono, monospace; font-size:0.7rem; color:#334155; text-align:center; margin-top:3rem;">'
    'Developed by Satyam Kumar</p>',
    unsafe_allow_html=True
)

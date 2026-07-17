import os
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

st.set_page_config(page_title="UTK Face Age & Gender Predictor", page_icon="👤", layout="wide")

# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "age_gender_model.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

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
        return img_rgb[max(0,y-margin):y+h+margin, max(0,x-margin):x+w+margin]
    return img_rgb

def preprocess(cropped):
    if cropped is None: return None
    resized = cv2.resize(cropped, (224, 224))
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR).astype(np.float32)
    bgr[..., 0] -= 103.939
    bgr[..., 1] -= 116.779
    bgr[..., 2] -= 123.68
    return np.expand_dims(bgr, 0)

# ====================== UI ======================
st.title("👤 UTK Face Age & Gender Predictor")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("⚙️ Controls")
    max_age = st.slider("Max Age", 1, 116, 116)
    bias = st.slider("Fine Tune", -15, 15, 0, step=1)

with col2:
    uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        uploaded_file.seek(0)
        face = crop_face(uploaded_file)
        
        if face is not None:
            left, right = st.columns(2)
            with left:
                st.image(face, use_container_width=True, caption="Cropped Face")
            
            with right:
                with st.spinner("Predicting..."):
                    tensor = preprocess(face)
                    preds = model.predict(tensor, verbose=0)

                    if isinstance(preds, list):
                        age_raw = float(preds[0][0][0])
                        gender_raw = float(preds[1][0][0])
                    else:
                        flat = np.array(preds).flatten()
                        age_raw = float(flat[0])
                        gender_raw = float(flat[1]) if len(flat)>1 else 0.5

                    # Z-score + Calibration
                    predicted_age = int(round(age_raw * 19.924 + 33.246)) + bias
                    predicted_age = max(1, min(predicted_age, max_age))

                    gender = "Female" if gender_raw >= 0.5 else "Male"

                    st.success(f"**Gender:** {gender}")
                    st.success(f"**Estimated Age:** {predicted_age} years")
                    st.caption(f"Raw: {age_raw:.4f}")

st.markdown("---")
st.markdown("<center><p style='color: gray;'>Developed by Satyam Kumar</p></center>", unsafe_allow_html=True)

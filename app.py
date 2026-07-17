import os
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

st.set_page_config(page_title="UTK Face Age & Gender Predictor", page_icon="👤", layout="wide")

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "age_gender_model.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_my_model()

# --- FACE & PREPROCESS ---
def crop_face(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not decode image.")
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

def preprocess_image(cropped):
    if cropped is None: return None
    resized = cv2.resize(cropped, (224, 224))
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR).astype(np.float32)
    bgr[..., 0] -= 103.939
    bgr[..., 1] -= 116.779
    bgr[..., 2] -= 123.68
    return np.expand_dims(bgr, 0)

# --- UI ---
st.title("👤 UTK Face Age & Gender Predictor")
st.markdown("**Debug Mode** - Help me calibrate")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("⚙️ Controls")
    max_age = st.slider("Max Age", 1, 116, 116)
    global_bias = st.slider("Global Bias", -30, 30, 0, step=1)
    st.markdown("---")
    st.markdown("**Developer**: Satyam Kumar")

with col2:
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        uploaded_file.seek(0)
        face = crop_face(uploaded_file)
        
        if face is not None:
            left, right = st.columns(2)
            with left:
                st.image(face, use_container_width=True, caption="Cropped Face")
            
            with right:
                with st.spinner("Predicting..."):
                    input_tensor = preprocess_image(face)
                    predictions = model.predict(input_tensor, verbose=0)

                    if isinstance(predictions, list) and len(predictions) == 2:
                        age_raw = float(predictions[0][0][0])
                        gender_raw = float(predictions[1][0][0])
                    else:
                        flat = np.asarray(predictions).flatten()
                        age_raw = float(flat[0])
                        gender_raw = float(flat[1]) if len(flat) > 1 else 0.5

                    # Z-score (your current method)
                    AGE_MEAN = 33.24633554782242
                    AGE_STD = 19.924041256760226
                    z_age = (age_raw * AGE_STD) + AGE_MEAN

                    st.success(f"**Gender:** {'Female' if gender_raw >= 0.5 else 'Male'}")
                    st.success(f"**Z-Score Age:** {int(round(z_age))} years")
                    st.caption(f"Raw Model Output: {age_raw:.5f}")
                    st.caption(f"After Z-Score: {z_age:.2f}")

                    # Alternative simple scaling (for comparison)
                    simple_age = int(round(age_raw * 80))   # fallback
                    st.caption(f"Simple Scaling (80x): {simple_age} years")

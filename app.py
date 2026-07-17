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
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_my_model()

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
        margin = int(0.32 * w)
        return img_rgb[max(0, y-margin):y+h+margin, max(0, x-margin):x+w+margin]
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

col_tune, col_main = st.columns([1, 3])

with col_tune:
    st.header("⚙️ Controls")
    max_age = st.slider("Maximum Age", 1, 116, 116)
    global_bias = st.slider("Global Fine-tune", -12, 12, 0, step=1)
    st.markdown("---")
    st.markdown("**Developer**: Satyam Kumar")

with col_main:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        uploaded_file.seek(0)
        cropped_face = crop_face(uploaded_file)

        if cropped_face is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📸 Cropped Face")
                st.image(cropped_face, use_container_width=True)

            with c2:
                st.subheader("🔮 Model Prediction")
                with st.spinner("Analyzing..."):
                    try:
                        input_tensor = preprocess_image(cropped_face)
                        predictions = model.predict(input_tensor, verbose=0)

                        if isinstance(predictions, list) and len(predictions) == 2:
                            age_raw = float(predictions[0][0][0])
                            gender_raw = float(predictions[1][0][0])
                        else:
                            flat = np.asarray(predictions).flatten()
                            age_raw = float(flat[0])
                            gender_raw = float(flat[1]) if len(flat) > 1 else 0.5

                        # Z-score + Custom Calibration
                        AGE_MEAN = 33.246
                        AGE_STD = 19.924
                        z_age = (age_raw * AGE_STD) + AGE_MEAN

                        # Custom correction for better real-world performance
                        if age_raw < 0.1:
                            final_age = z_age - 8          # Reduce overestimation on children/young
                        elif age_raw < 0.5:
                            final_age = z_age - 4
                        else:
                            final_age = z_age - 2          # Slight correction on adults

                        predicted_age = int(round(final_age)) + global_bias
                        predicted_age = max(1, min(predicted_age, max_age))

                        gender_label = "Female" if gender_raw >= 0.5 else "Male"
                        gender_conf = max(gender_raw, 1.0 - gender_raw)

                        st.success(f"**Gender:** {gender_label}")
                        st.progress(gender_conf)
                        st.caption(f"Confidence: {gender_conf*100:.1f}%")
                        st.success(f"**Estimated Age:** {predicted_age} years")
                        st.caption(f"Raw output: {age_raw:.4f} | Z-Age: {z_age:.1f}")

                    except Exception as e:
                        st.error("Prediction error")
                        st.exception(e)

st.markdown("---")
st.markdown("<center><p style='color: gray;'>Developed with ❤️ by Satyam Kumar</p></center>", unsafe_allow_html=True)

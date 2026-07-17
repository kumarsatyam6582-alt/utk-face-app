import os
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="UTK Face Age & Gender Predictor",
    page_icon="👤",
    layout="wide"
)

# --- CONFIG ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "age_gender_model.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# --- MODEL LOADING with Better Error Handling ---
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: `{MODEL_PATH}`")
        st.info("Please upload your `age_gender_model.keras` file into the `model/` folder.")
        st.stop()
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error("❌ Failed to load model. The file may be corrupted or incompatible.")
        st.exception(e)
        st.stop()

model = load_my_model()

# --- FACE DETECTION ---
def crop_face(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("❌ Could not decode image.")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if not os.path.exists(CASCADE_PATH):
        return img_rgb

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        margin = int(0.32 * w)
        y1 = max(0, y - margin)
        y2 = min(img_rgb.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(img_rgb.shape[1], x + w + margin)
        return img_rgb[y1:y2, x1:x2]
    return img_rgb

# --- PREPROCESSING ---
def preprocess_image(cropped_img):
    if cropped_img is None:
        return None
    resized = cv2.resize(cropped_img, (IMG_WIDTH, IMG_HEIGHT))
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR).astype(np.float32)
    bgr[..., 0] -= 103.939
    bgr[..., 1] -= 116.779
    bgr[..., 2] -= 123.68
    return np.expand_dims(bgr, axis=0)

# --- UI ---
st.title("👤 UTK Face Age & Gender Predictor")
st.markdown("**Optimized for All Age Groups**")

col_tune, col_main = st.columns([1, 3])

with col_tune:
    st.header("⚙️ Controls")
    max_age = st.slider("Maximum Age", 1, 116, 116)
    global_bias = st.slider("Global Fine-tune", -15, 15, 0, step=1)
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
                with st.spinner("Analyzing face..."):
                    try:
                        input_tensor = preprocess_image(cropped_face)
                        predictions = model.predict(input_tensor, verbose=0)

                        if isinstance(predictions, list) and len(predictions) == 2:
                            age_raw = float(predictions[0][0][0])
                            gender_raw = float(predictions[1][0][0])
                        else:
                            preds = np.asarray(predictions).flatten()
                            age_raw = float(preds[0])
                            gender_raw = float(preds[1]) if len(preds) > 1 else 0.5

                        # Optimized Scaling
                        raw = age_raw
                        if raw < -0.4:
                            predicted_age = int(round((raw + 1.22) * 85))
                        elif raw < 0.05:
                            predicted_age = int(round(raw * 520))
                        elif raw < 0.4:
                            predicted_age = int(round(raw * 110))
                        elif raw < 1.0:
                            predicted_age = int(round(raw * 85))
                        else:
                            predicted_age = int(round(raw * 3.2))

                        predicted_age = predicted_age + global_bias
                        predicted_age = max(1, min(predicted_age, max_age))

                        gender_label = "Female" if gender_raw >= 0.5 else "Male"
                        gender_conf = max(gender_raw, 1.0 - gender_raw)

                        st.success(f"**Gender:** {gender_label}")
                        st.progress(gender_conf)
                        st.caption(f"Confidence: {gender_conf*100:.1f}%")

                        st.success(f"**Estimated Age:** {predicted_age} years")
                        st.caption(f"Raw output: {age_raw:.4f}")

                    except Exception as e:
                        st.error("❌ Prediction error")
                        st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    "<center><p style='color: gray;'>Developed with ❤️ by Satyam Kumar</p></center>",
    unsafe_allow_html=True
)

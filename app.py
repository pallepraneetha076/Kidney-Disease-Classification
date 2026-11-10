import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os

from PIL import Image
from tensorflow.keras.models import Model

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Kidney Disease Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("Kidney Disease Classification")
st.write(
    "Deep learning-based classification of kidney CT scan images "
    "using VGG16 and Grad-CAM interpretability."
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_URL = "https://github.com/pallepraneetha076/Kidney-Disease-Classification/releases/download/v1.0/kidney_model.keras"

MODEL_PATH = os.path.join(
    tempfile.gettempdir(),
    "kidney_model.keras"
)

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model..."):
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    return MODEL_PATH

@st.cache_resource
def load_trained_model():
    model_path = download_model()
    return tf.keras.models.load_model(model_path)

try:
    model = load_trained_model()
    vgg_model = model.get_layer("vgg16")

    feature_extractor = Model(
        inputs=vgg_model.input,
        outputs=vgg_model.get_layer("block5_conv3").output
    )

    grad_input = tf.keras.Input(shape=(224, 224, 3))

    x = model.get_layer("data_augmentation")(
        grad_input,
        training=False
    )

    x = tf.keras.applications.vgg16.preprocess_input(x)

    conv_outputs = feature_extractor(x, training=False)

    x = model.get_layer("global_average_pooling2d")(conv_outputs)
    x = model.get_layer("dense")(x)
    x = model.get_layer("dropout")(x, training=False)
    predictions = model.get_layer("dense_1")(x)

    grad_model = Model(
        inputs=grad_input,
        outputs=[conv_outputs, predictions]
    )

except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# --------------------------------------------------
# GRAD-CAM FUNCTION
# --------------------------------------------------

def generate_gradcam(image_array):

    image_batch = np.expand_dims(
        image_array.astype(np.float32),
        axis=0
    )

    with tf.GradientTape() as tape:

        conv_features, predictions = grad_model(
            image_batch,
            training=False
        )

        tumor_score = predictions[:, 0]

    gradients = tape.gradient(
        tumor_score,
        conv_features
    )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    heatmap = tf.reduce_sum(
        conv_features * weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap[0]

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    original_bgr = cv2.cvtColor(
        np.uint8(image_array),
        cv2.COLOR_RGB2BGR
    )

    overlay = cv2.addWeighted(
        original_bgr, 0.6,
        heatmap_color, 0.4,
        0
    )

    return heatmap_color, overlay

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.subheader("Upload Kidney CT Scan")

uploaded_file = st.file_uploader(
    "Choose a CT scan image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)

    st.subheader("Classification Result")

    with st.spinner("Analyzing image..."):

        input_batch = np.expand_dims(image_array, axis=0)

        probability = float(
            model.predict(input_batch, verbose=0)[0][0]
        )

        predicted_class = (
            "Tumor" if probability >= 0.5 else "Normal"
        )

        heatmap, overlay = generate_gradcam(image_array)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded CT Scan",
            use_container_width=True
        )

    with col2:
        st.metric("Predicted Class", predicted_class)
        st.metric(
            "Tumor Probability",
            f"{probability * 100:.2f}%"
        )

    st.subheader("Grad-CAM Interpretability")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    with col2:
        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            caption="Grad-CAM Overlay",
            use_container_width=True
        )

    st.info(
        "Grad-CAM highlights image regions that influenced the "
        "model prediction. It does not confirm a tumor or provide "
        "a medical diagnosis."
    )

st.divider()

st.caption(
    "Academic Deep Learning Project | VGG16 Transfer Learning | Grad-CAM"
)

st.warning(
    "This application is an educational prototype, not a clinical "
    "diagnostic system. Do not use its predictions for medical decisions."
)

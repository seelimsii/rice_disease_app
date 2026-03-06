import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from google import genai
from PIL import Image

# 1. Setup & Load Model
st.set_page_config(page_title="Rice Disease Guardian", layout="wide")
st.title("🌾 Rice Disease AI Diagnostic Tool")

@st.cache_resource
def load_all_resources():
    model = tf.keras.models.load_model("rice_disease_model.h5")
    with open("class_indices.json", "r") as f:
        indices = json.load(f)
    # Reverse the dict to get {index: name}
    class_names = {v: k for k, v in indices.items()}
    return model, class_names

model, class_names = load_all_resources()
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# 2. Grad-CAM Function
def get_gradcam(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 3. UI: Upload Section
uploaded_file = st.file_uploader("Upload a photo of a rice leaf...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Preprocessing
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_batch)
    pred_idx = np.argmax(preds)
    disease = class_names[pred_idx]
    confidence = np.max(preds) * 100

    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.metric("Detected Disease", disease)
        st.write(f"Confidence: {confidence:.2f}%")

    with col2:
        heatmap = get_gradcam(img_batch, model, "conv5_block16_concat")
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        st.image(heatmap_color, caption="AI Focus Area (Grad-CAM)", use_column_width=True)

    # 4. Gemini Advice
    st.subheader("🤖 AI Agricultural Advice")
    with st.spinner("Consulting Gemini..."):
        prompt = f"A rice plant leaf is diagnosed with {disease}. Explain in short, simple, farmer-friendly terms: what it is, causes, treatment, and if the leaf should be removed."
        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        st.write(response.text)
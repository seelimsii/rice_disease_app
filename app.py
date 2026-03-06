import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
import os
from google import genai
from PIL import Image
import urllib.request

# 1. Page Configuration
st.set_page_config(page_title="Rice Disease Guardian", layout="wide", page_icon="🌾")

# Custom CSS to make it look modern
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🌾 Rice Disease AI Diagnostic Tool")
st.write("Upload a clear photo of a rice leaf to identify diseases and get AI-powered treatment advice.")

# 2. Resource Loading (Cached for Speed)
@st.cache_resource
def load_all_resources():
    model_path = "rice_disease_model.h5"
    
    # DOWNLOAD THE MODEL IF MISSING
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI Model from GitHub (this happens once)..."):
            # PASTE YOUR DIRECT LINK BELOW
            url = "https://github.com/seelimsii/rice_disease_app/releases/download/v1.0.0/rice_disease_model.h5"
            urllib.request.urlretrieve(url, model_path)
            st.success("Model downloaded!")

    # Now load it normally
    model = tf.keras.models.load_model(model_path)
    
    with open("class_indices.json", "r") as f:
        indices = json.load(f)
    
    class_names = {int(v): k for k, v in indices.items()}
    return model, class_names

model, class_names = load_all_resources()

# Initialize Gemini Client safely
try:
    # Check if secret exists before initializing
    if "GEMINI_API_KEY" in st.secrets:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    else:
        st.warning("⚠️ Gemini API Key not found in Secrets. AI advice will be disabled.")
        client = None
except Exception as e:
    st.error(f"Error connecting to AI service: {e}")
    client = None

# 3. Grad-CAM Logic
def get_gradcam(img_batch, model, last_conv_layer_name):
    try:
        # 1. Create a model that maps input to the activations of the last conv layer as well as the output
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs], 
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            # We must ensure img_batch is a float32 tensor
            img_batch_tensor = tf.cast(img_batch, tf.float32)
            conv_outputs, predictions = grad_model(img_batch_tensor)
            
            # Find the index of the winning class
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        # 2. Extract gradients for the winning class with respect to the output feature map
        grads = tape.gradient(loss, conv_outputs)
        
        # 3. Vector of shape (1024,), where each entry is the mean intensity of the gradient over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 4. Multiply each channel in the feature map by "how important it is"
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # 5. Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")
        return None

# 4. User Interface
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file and model and class_names:
    # --- IMAGE PROCESSING ---
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # --- PREDICTION ---
    with st.spinner("Analyzing leaf patterns..."):
        preds = model.predict(img_batch, verbose=0)
        pred_idx = np.argmax(preds)
        disease = class_names[pred_idx]
        confidence = np.max(preds) * 100

    # --- RESULTS DISPLAY ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img, caption="Uploaded Image", width="stretch")
        st.metric("Detected Status", disease, f"{confidence:.1f}% Match")

    with col2:
        # We use the specific DenseNet layer you identified earlier
        heatmap = get_gradcam(img_batch, model, "relu")
        
        if heatmap is not None:
            # Colorize and overlay
            heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # Convert BGR to RGB for Streamlit
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            # Overlay original with heatmap (50% transparency)
            overlayed_img = cv2.addWeighted(np.array(img), 0.6, heatmap_color_rgb, 0.4, 0)
            st.image(overlayed_img, caption="AI Diagnosis Areas (Heatmap)", use_container_width=True)
        else:
            st.info("Visual explanation (Heatmap) not available for this model architecture.")

    # --- AI ADVICE SECTION ---
    st.divider()
    if client:
        st.subheader(f"🤖 AI Expert Advice: {disease}")
        if confidence < 60:
            st.warning("Note: The AI confidence is low. Please ensure the photo is clear and verify with a local expert.")
        
        with st.spinner("Generating farmer-friendly guide..."):
            prompt = f"""
            The rice plant leaf is diagnosed with {disease} (Confidence: {confidence:.1f}%).
            Explain in short, simple, farmer-friendly bullet points:
            1. What is this disease?
            2. Primary cause?
            3. Recommended organic and chemical treatments.
            4. Should the leaf be removed?
            5. One simple prevention tip.
            """
            try:
                response = client.models.generate_content(
                    model="	gemini-3-flash-preview", # Using 1.5-Flash for better stability in production
                    contents=prompt
                )
                st.markdown(response.text)
            except Exception as e:
                st.error("Could not fetch advice. Please check your internet connection or API quota.")
    else:
        st.info("Connect your Gemini API key in Settings > Secrets to see treatment advice.")

elif not uploaded_file:
    st.info("Please upload an image to start the diagnosis.")

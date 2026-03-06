import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from google import genai
from PIL import Image
import urllib.request

# 1. Page Configuration
st.set_page_config(page_title="Rice Disease Guardian", layout="wide", page_icon="🌾")

# Custom CSS
st.markdown("""
<style>
.main { background-color: #f5f7f9; }
.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Rice Disease AI Diagnostic Tool")
st.write("Upload a clear photo of a rice leaf to identify diseases and get AI-powered treatment advice.")

# 2. Load Model and Class Labels
@st.cache_resource
def load_all_resources():

    model_path = "rice_disease_model.h5"

    if not os.path.exists(model_path):
        with st.spinner("Downloading AI Model from GitHub (this happens once)..."):
            url = "https://github.com/seelimsii/rice_disease_app/releases/download/v1.0.0/rice_disease_model.h5"
            urllib.request.urlretrieve(url, model_path)
            st.success("Model downloaded!")

    model = tf.keras.models.load_model(model_path)

    with open("class_indices.json", "r") as f:
        indices = json.load(f)

    class_names = {int(v): k for k, v in indices.items()}

    return model, class_names


model, class_names = load_all_resources()

# 3. Initialize Gemini Client
try:
    if "GEMINI_API_KEY" in st.secrets:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    else:
        st.warning("⚠️ Gemini API Key not found in Secrets. AI advice will be disabled.")
        client = None
except Exception as e:
    st.error(f"Error connecting to AI service: {e}")
    client = None


# 4. Upload Image
uploaded_file = st.file_uploader("Choose a rice leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file and model and class_names:

    # Image preprocessing
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))

    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Analyzing leaf patterns..."):
        preds = model.predict(img_batch, verbose=0)

    pred_idx = np.argmax(preds)
    disease = class_names[pred_idx]
    confidence = np.max(preds) * 100

    # Display Results
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

    with col2:
        st.metric("Detected Disease", disease)
        st.metric("Confidence", f"{confidence:.2f}%")

    # Optional: Probability chart
    st.subheader("Prediction Probabilities")

    prob_dict = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
    st.bar_chart(prob_dict)

    # AI Advice Section
    st.divider()

    if client:

        st.subheader(f"🤖 AI Expert Advice for {disease}")

        if confidence < 60:
            st.warning("Model confidence is low. Ensure the image is clear and consult a local agricultural expert.")

        with st.spinner("Generating treatment advice..."):

            prompt = f"""
The rice leaf has been diagnosed with {disease} (Confidence: {confidence:.1f}%).

Explain in short farmer-friendly bullet points:

1. What is this disease?
2. What causes it?
3. Organic treatment options.
4. Chemical treatment options.
5. Should the infected leaf be removed?
6. One simple prevention tip.
"""

            try:

                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=prompt
                )

                st.markdown(response.text)

            except Exception:
                st.error("Could not generate advice. Check API key or quota.")

    else:
        st.info("Add your Gemini API key in Streamlit Secrets to enable AI treatment advice.")


elif not uploaded_file:

    st.info("Please upload an image of a rice leaf to begin diagnosis.")

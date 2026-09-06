"""
Streamlit Web Application: Real-Time Handwritten Digit Recognizer
Enables users to:
1. Snap a photo of a digit using their webcam/phone camera.
2. Draw a digit directly on an interactive canvas.
3. Upload an existing digit photo or image.
"""

import os
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

from preprocess_utils import preprocess_digit

MODEL_PATH = "mnist_cnn_model.keras"

# Set Streamlit page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .pred-card {
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .pred-digit {
        font-size: 5rem;
        font-weight: 900;
        line-height: 1;
        margin: 10px 0;
    }
    .pred-conf {
        font-size: 1.3rem;
        font-weight: 600;
        opacity: 0.95;
    }
    .metric-badge {
        display: inline-block;
        background: rgba(255,255,255,0.25);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cnn_model(path):
    if not os.path.exists(path):
        return None
    model = tf.keras.models.load_model(path)
    return model

model = load_cnn_model(MODEL_PATH)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", caption="MNIST Handwritten Digits")
    st.title("About the Model")
    st.markdown("""
    **Architecture:**
    - `Conv2D (32 filters, 3x3)` + `ReLU`
    - `MaxPooling2D (2x2)`
    - `Conv2D (64 filters, 3x3)` + `ReLU`
    - `MaxPooling2D (2x2)`
    - `Dense (128 units)` + `Dropout(0.5)`
    - `Dense (10 units, Softmax)`

    **Performance:**
    - Test Accuracy: **99.09%**
    - Dataset: MNIST (70,000 digits)
    """)
    st.divider()
    st.subheader("Settings")
    thicken = st.checkbox("Thicken Pen Strokes", value=True, help="Enhances thin ballpoint pen strokes to match MNIST marker thickness")

# ==============================================================================
# MAIN PAGE
# ==============================================================================
st.markdown('<div class="main-title">🔢 Handwritten Digit Recognizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by Convolutional Neural Network (CNN) • Camera & Canvas Input</div>', unsafe_allow_html=True)

if model is None:
    st.error(f"⚠️ Model file `{MODEL_PATH}` not found! Please train the model first by running `python train_model.py`.")
    st.stop()

# Input Options (Tabs)
tab1, tab2, tab3 = st.tabs(["📸 Camera Snap", "🖌️ Interactive Drawing Canvas", "📁 Upload Image"])

image_to_process = None

with tab1:
    st.write("Hold a piece of paper with a handwritten digit in front of your camera and snap a picture:")
    camera_photo = st.camera_input("Take a photo of your digit")
    if camera_photo is not None:
        pil_img = Image.open(camera_photo)
        image_to_process = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

with tab2:
    st.write("Draw a single digit (0-9) inside the black box below:")
    try:
        from streamlit_drawable_canvas import st_canvas
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="digit_canvas",
        )
        
        if canvas_result.image_data is not None:
            # Check if user has drawn anything
            img_data = canvas_result.image_data
            if np.sum(img_data[:, :, 0]) > 0:
                # User drew on black background with white stroke
                # Convert RGBA to BGR
                image_to_process = cv2.cvtColor(img_data.astype('uint8'), cv2.COLOR_RGBA2BGR)
    except Exception as e:
        st.info("Interactive canvas requires `streamlit-drawable-canvas`. Use Camera or Upload tabs above.")

with tab3:
    st.write("Upload an image containing a handwritten number (JPG, PNG):")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        image_to_process = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ==============================================================================
# INFERENCE & PREDICTION DISPLAY
# ==============================================================================
if image_to_process is not None:
    st.divider()
    
    # Preprocess the image
    input_tensor, preprocessed_28, is_empty = preprocess_digit(image_to_process, thicken_strokes=thicken)
    
    if is_empty:
        st.warning("⚠️ No clear digit stroke detected. Please draw or show a clearer number.")
    else:
        # Run CNN Model Inference
        probs = model.predict(input_tensor, verbose=0)[0]
        pred_digit = int(np.argmax(probs))
        conf_pct = float(probs[pred_digit]) * 100.0

        col_left, col_right = st.columns([1, 1.4])

        with col_left:
            # Prediction Card
            st.markdown(f"""
            <div class="pred-card">
                <span class="metric-badge">PREDICTED DIGIT</span>
                <div class="pred-digit">{pred_digit}</div>
                <div class="pred-conf">{conf_pct:.2f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)

            # Top 3 Candidates
            top3_idx = np.argsort(probs)[-3:][::-1]
            st.write("**Top 3 Candidates:**")
            cand_cols = st.columns(3)
            for rank, (c_col, idx) in enumerate(zip(cand_cols, top3_idx), 1):
                with c_col:
                    st.metric(label=f"#{rank} Digit: {idx}", value=f"{probs[idx]*100:.1f}%")

            st.write("---")
            # Side-by-side view of input vs CNN preprocessed 28x28
            vis_c1, vis_c2 = st.columns(2)
            with vis_c1:
                st.caption("📷 Original Input")
                st.image(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB), use_container_width=True)
            with vis_c2:
                st.caption("🧠 What CNN Sees (28x28 Centered)")
                st.image(preprocessed_28, use_container_width=True)

        with col_right:
            st.subheader("📊 Probability Distribution (0–9)")
            
            # Prepare dataframe for chart
            df_probs = pd.DataFrame({
                "Digit": [str(i) for i in range(10)],
                "Probability (%)": [float(p * 100) for p in probs]
            })
            
            st.bar_chart(df_probs.set_index("Digit"), color="#1E88E5", height=380)

            # Detailed table
            with st.expander("🔍 View Raw Softmax Probabilities"):
                st.dataframe(df_probs.style.format({"Probability (%)": "{:.3f}%"}), use_container_width=True)
else:
    st.info("👆 Use the tabs above to snap a photo with your camera, draw on the canvas, or upload an image to see live predictions!")

import time

import streamlit as st
from PIL import Image

from helpers import BackgroundRemoval


# Load model
@st.cache_resource
def load_model(model_path, device_inference="cpu"):
    model = BackgroundRemoval(model_path, device_inference.lower())
    return model


# Set page config
st.set_page_config(
    page_title="Background Removal",
    page_icon="🏡",
)


# Main page
st.title("Background Removal")
st.write(
    """
      Background removal refers to the process of separating and eliminating the background of an image or video, leaving only the subject or foreground visible. This technique is widely used in various applications, such as photo editing, video production, and computer vision. The result can be used to replace the background with a new one, isolate the subject for further processing, or simply improve the visual quality of the content. The implementation of background removal can vary, from simple techniques like chroma keying to more advanced methods like deep learning algorithms.
"""
)
st.markdown("  ")
device_inference = st.sidebar.selectbox("Select device", ("CPU", "CUDA"))
path_model = st.sidebar.text_input(
    "Path to model", "./saved_model_background_removal.onnx"
)

# Run load model
model = load_model(path_model, device_inference)
uploaded_file = st.file_uploader(
    "Upload image file", type=["jpg", "jpeg", "png", "bmp", "tiff"]
)
if uploaded_file is not None:
    uploaded_file = Image.open(uploaded_file).convert("RGB")
    st.markdown("  ")
    st.write("Source Image")
    st.image(uploaded_file)

    predict_button = st.button("Remove background")
    st.markdown("  ")

    if predict_button:
        with st.spinner("Wait for it..."):
            start_time = time.perf_counter()
            segmented_image, mask_image = model.predict(uploaded_file)
            st.write(
                f"Inference time: {(time.perf_counter() - start_time):.3f} seconds"
            )
            col1, col2 = st.columns(2)
            col1.image(segmented_image)
            col2.image(mask_image)

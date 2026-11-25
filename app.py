from io import BytesIO

import streamlit as st
import torch
from PIL import Image, ImageOps

from script3 import CLASSES, DEVICE, MODEL_FILE, load_trained_model, normalize, open_xray


@st.cache_resource(show_spinner=False)
def cached_model(checkpoint_version):
    # Streamlit reloads the weights when training replaces the checkpoint.
    return load_trained_model()


def prepare_upload(contents):
    with Image.open(BytesIO(contents)) as original:
        if original.format not in {"JPEG", "PNG"}:
            raise ValueError("Choose a PNG or JPEG image.")
        if original.width * original.height > 25_000_000:
            raise ValueError("Choose an image with 25 megapixels or fewer.")
        if original.mode not in {"L", "P", "RGB", "RGBA"}:
            raise ValueError("Choose an 8-bit grayscale, palette, or RGB image.")
        preview = ImageOps.exif_transpose(original).convert("L")
        preview.thumbnail((800, 800))

    # Use exactly the same resizing, padding and normalization as training.
    model_input = normalize(open_xray(BytesIO(contents))).unsqueeze(0)
    return preview, model_input


def main():
    st.set_page_config(page_title="Chest X-ray classifier", layout="centered")
    st.title("Chest X-ray classifier")
    st.write("Choose a frontal chest X-ray to see which of the three dataset labels the model predicts.")
    st.caption("Educational demo. The model assigns a label to every accepted image, including unrelated photos. Results are unsuitable for diagnosis.")

    if not MODEL_FILE.is_file():
        st.error("Missing models/best.pt. Place app.py beside script3.py in the project containing your trained model.")
        return

    uploaded = st.file_uploader("Choose an X-ray", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        return
    if uploaded.size > 25 * 1024 * 1024:
        st.error("Choose an image smaller than 25 MB.")
        return

    try:
        preview, model_input = prepare_upload(uploaded.getvalue())
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        st.error(f"Unable to read this image. {error}")
        return

    left, right = st.columns([3, 2])
    with left:
        st.image(preview, caption=uploaded.name)

    with right:
        with st.spinner("Classifying image…"):
            model = cached_model(MODEL_FILE.stat().st_mtime_ns)
            with torch.inference_mode():
                scores = model(model_input.to(DEVICE))
                label = CLASSES[scores.argmax(dim=1).item()]
        st.caption("Predicted dataset label")
        st.subheader(label.capitalize())
        st.caption("Model · ResNet-18")


if __name__ == "__main__":
    main()

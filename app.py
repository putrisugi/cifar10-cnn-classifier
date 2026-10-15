from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


CLASS_NAMES = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]
MODEL_PATH = Path(__file__).resolve().parent / "models" / "optimized_cnn.keras"

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #1d2935;
        --muted: #5b6875;
        --line: #d9e0e6;
        --accent: #315f7c;
    }
    [data-testid="stAppViewContainer"] {
        background: #f7f8f9;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: #f7f8f9; }
    .main .block-container {
        max-width: 1160px;
        padding: 1.7rem 1.65rem 1.5rem;
    }
    h1, h2, h3, p, span, label {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    h1, h2, h3 { color: var(--ink); }
    .app-header {
        margin: 0 0 1.25rem;
        padding: 0 0 .9rem;
        border-bottom: 1px solid var(--line);
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 650;
        letter-spacing: -.025em;
        line-height: 1.25;
    }
    .app-header p {
        margin: .28rem 0 0;
        color: var(--muted);
        font-size: .91rem;
        line-height: 1.45;
    }
    .section-heading {
        margin: 0 0 .65rem;
        color: var(--ink);
        font-size: 1rem;
        font-weight: 650;
        letter-spacing: -.01em;
    }
    [data-testid="stFileUploader"] {
        padding: .1rem;
        border: 1px dashed #aebbc6;
        border-radius: 5px;
        background: #fff;
    }
    [data-testid="stFileUploaderDropzone"] { border: 0; background: transparent; }
    [data-testid="stFileUploaderDropzone"] button {
        border: 1px solid #aebbc6;
        border-radius: 4px;
        color: #29465a;
        background: white;
    }
    [data-testid="stImage"] img { border-radius: 3px; }
    .result-summary {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        padding: .1rem 0 .8rem;
        border-bottom: 1px solid var(--line);
    }
    .field-label {
        margin: 0 0 .25rem;
        color: var(--muted);
        font-size: .78rem;
        font-weight: 550;
    }
    .prediction-class {
        margin: 0;
        color: var(--ink);
        font-size: 1.7rem;
        font-weight: 650;
        letter-spacing: -.03em;
        line-height: 1.2;
    }
    .confidence-value {
        min-width: 6rem;
        text-align: right;
    }
    .confidence-value strong {
        color: var(--accent);
        font-size: 1.15rem;
        font-weight: 650;
    }
    .minor-heading {
        margin: 1rem 0 .35rem;
        color: var(--ink);
        font-size: .88rem;
        font-weight: 650;
    }
    .rank-row {
        display: grid;
        grid-template-columns: 2rem minmax(0, 1fr) 4rem;
        align-items: center;
        gap: .55rem;
        padding: .42rem 0;
        border-bottom: 1px solid #e4e8ec;
        color: var(--ink);
        font-size: .88rem;
    }
    .rank-number { color: #687786; font-size: .76rem; font-variant-numeric: tabular-nums; }
    .rank-name { font-weight: 550; }
    .rank-score { color: #425b70; font-variant-numeric: tabular-nums; text-align: right; }
    .model-info {
        margin-top: 1.15rem;
        padding-top: .65rem;
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: .78rem;
        line-height: 1.5;
    }
    .model-info strong { color: #384957; font-weight: 650; }
    @media (max-width: 700px) {
        .main .block-container { padding: 1.2rem 1rem 1.25rem; }
        .app-header { margin-bottom: 1rem; }
        .app-header h1 { font-size: 1.35rem; }
        .prediction-class { font-size: 1.45rem; }
        [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the classifier…")
def load_model(model_path: str) -> Any:
    """Load the existing trained model once for the Streamlit process."""
    import tensorflow as tf

    return tf.keras.models.load_model(model_path, compile=False)


def prepare_image(image_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
    """Decode an uploaded image and prepare one normalized CIFAR-sized batch."""
    with Image.open(BytesIO(image_bytes)) as source:
        if source.format not in {"JPEG", "PNG"}:
            raise ValueError("Please upload a JPG, JPEG, or PNG image.")
        original = ImageOps.exif_transpose(source).convert("RGB")

    resized = original.resize((32, 32), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    batch = np.expand_dims(pixels, axis=0)
    return original, batch


def predict(batch: np.ndarray) -> np.ndarray:
    """Run inference and return a validated ten-class probability vector."""
    model = load_model(str(MODEL_PATH))
    raw_output = np.asarray(model.predict(batch, verbose=0), dtype=np.float64)
    if raw_output.shape != (1, len(CLASS_NAMES)):
        raise ValueError("The model returned an unexpected output shape.")

    probabilities = raw_output[0]
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("The model returned invalid probability values.")

    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("The model returned an empty probability distribution.")
    return probabilities / total


st.markdown(
    '<div class="app-header"><h1>CIFAR-10 Image Classifier</h1><p>Classify an image with the trained convolutional neural network.</p></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.08], gap="large")
uploaded_file = None
original_image = None
image_batch = None
image_error = None

with left:
    st.markdown('<div class="section-heading">Input image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        help="The image is converted to RGB and resized to 32 × 32 pixels for inference.",
    )
    if uploaded_file is not None:
        try:
            original_image, image_batch = prepare_image(uploaded_file.getvalue())
            st.image(original_image, caption="Uploaded image", use_container_width=True)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            image_error = exc
            st.error("We couldn’t read that image. Please choose a valid JPG, JPEG, or PNG file.")
        except Exception as exc:  # Keep unexpected decoder errors inside the app UI.
            image_error = exc
            st.error("Something went wrong while preparing this image. Please try another file.")

with right:
    st.markdown('<div class="section-heading">Prediction</div>', unsafe_allow_html=True)
    if uploaded_file is None:
        st.caption("Upload an image to see the predicted class and probability scores.")
    elif image_error is not None or image_batch is None:
        st.caption("Choose a readable JPG, JPEG, or PNG file to continue.")
    elif not MODEL_PATH.is_file():
        st.error(f"The model file was not found at `{MODEL_PATH.relative_to(Path(__file__).resolve().parent)}`. Add the trained model at that path and restart the app.")
    else:
        try:
            probabilities = predict(image_batch)
        except Exception as exc:
            st.error("The classifier could not make a prediction. Check the model file and TensorFlow installation, then try again.")
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")
        else:
            ranked_indices = np.argsort(probabilities)[::-1]
            best_index = int(ranked_indices[0])

            st.markdown(
                f'<div class="result-summary"><div><p class="field-label">Predicted class</p><p class="prediction-class">{CLASS_NAMES[best_index]}</p></div><div class="confidence-value"><p class="field-label">Confidence</p><strong>{probabilities[best_index] * 100:.1f}%</strong></div></div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="minor-heading">Top predictions</div>', unsafe_allow_html=True)
            for rank, class_index in enumerate(ranked_indices[:3], start=1):
                score = probabilities[int(class_index)] * 100
                st.markdown(
                    f'<div class="rank-row"><span class="rank-number">0{rank}</span><span class="rank-name">{CLASS_NAMES[int(class_index)]}</span><span class="rank-score">{score:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="minor-heading">Class probabilities</div>', unsafe_allow_html=True)
            chart_data = pd.DataFrame(
                {"Probability (%)": probabilities * 100},
                index=CLASS_NAMES,
            )
            st.bar_chart(chart_data, horizontal=True, height=300, use_container_width=True)

st.markdown(
    '<div class="model-info"><strong>Model information</strong> &nbsp; CIFAR-10 CNN · reported test accuracy: 77.42%. Real-world photos may differ from the training images; confidence is an estimate, not a guarantee.</div>',
    unsafe_allow_html=True,
)

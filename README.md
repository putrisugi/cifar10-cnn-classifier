---
title: CIFAR-10 Image Classifier
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# CIFAR-10 Image Classifier

A Streamlit app that uses the existing CNN in `models/optimized_cnn.keras` to classify images into the ten CIFAR-10 classes. The model is loaded for inference only; it is not retrained or changed by this app.

## Run locally

Use Python 3.11, then install the dependencies and start Streamlit:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Run with Docker

```bash
docker build -t cifar10-image-classifier .
docker run --rm -p 8501:8501 cifar10-image-classifier
```

Then open `http://localhost:8501`.

## Deploy to Hugging Face Spaces

Create a new Space with the **Docker** SDK, then add this project’s `app.py`, `requirements.txt`, `Dockerfile`, README, and `models/optimized_cnn.keras` at the same paths. Push those files to the Space repository; Hugging Face builds the Docker image and serves the app on port 8501.

## Model notes

The network expects RGB images resized to 32 × 32 with pixel values scaled to `[0, 1]`. Its reported CIFAR-10 test accuracy is 77.42%. Since real-world images can differ from CIFAR-10, results may be less reliable outside that dataset. Confidence scores are model estimates and do not guarantee correctness.

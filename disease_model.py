"""Local CNN inference using the PlantVillage fine-tuned Hugging Face model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageOps


HF_MODEL_ID = "rarfileexe/Plant-Disease-Detector"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___Healthy",
    "Blueberry___Healthy",
    "Cherry___Powdery_mildew",
    "Cherry___Healthy",
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___Healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___Healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___Healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___Healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy",
    "Raspberry___Healthy",
    "Soybean___Healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___Healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Healthy",
]


class DiseaseModelError(RuntimeError):
    """Raised for dependency, download, or inference errors."""


@dataclass(frozen=True)
class DiseasePrediction:
    class_name: str
    display_name: str
    crop_name: str
    disease_name: str
    confidence: float


def load_plant_disease_model() -> Any:
    """Load the public Keras model from Hugging Face on first use."""
    os.environ["KERAS_BACKEND"] = "tensorflow"
    try:
        import keras
        import tensorflow  # noqa: F401  # Ensure the configured Keras backend is installed.
    except ImportError as exc:
        raise DiseaseModelError(
            "Disease diagnosis needs TensorFlow/Keras. Install the optional dependencies with "
            "`python -m pip install -r requirements-disease.txt`."
        ) from exc

    try:
        model = keras.saving.load_model(f"hf://{HF_MODEL_ID}")
    except Exception as exc:
        raise DiseaseModelError(
            "Could not download or load the Hugging Face CNN. Check the internet connection and "
            "that the model repository is reachable, then retry."
        ) from exc

    output_shape = getattr(model, "output_shape", None)
    if output_shape and output_shape[-1] != len(CLASS_NAMES):
        raise DiseaseModelError(
            f"The model returns {output_shape[-1]} classes but {len(CLASS_NAMES)} labels are configured."
        )
    return model


def prepare_image(image: Image.Image) -> np.ndarray:
    """Apply the RGB/224x224 preprocessing used by the model card."""
    image = ImageOps.exif_transpose(image).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    # The model card says pixel rescaling is included in the saved Keras model.
    return np.asarray(image, dtype=np.float32)[np.newaxis, ...]


def _decode_label(label: str) -> tuple[str, str, str]:
    crop, _, condition = label.partition("___")
    crop_display = crop.replace(",_", " ").replace("_", " ").strip()
    condition_display = condition.replace("_", " ").replace(",", " ").strip()
    if crop == "Orange" and "Haunglongbing" in condition:
        condition_display = "Citrus greening (huanglongbing)"
    if condition_display.casefold() == "healthy":
        display = f"{crop_display} · Healthy"
    else:
        display = f"{crop_display} · {condition_display}"
    return crop_display, condition_display, display


def predict_disease(model: Any, image: Image.Image, top_n: int = 3) -> list[DiseasePrediction]:
    """Return the model's highest-probability PlantVillage classes."""
    try:
        scores = np.asarray(model.predict(prepare_image(image), verbose=0), dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise DiseaseModelError("The CNN could not analyze this image. Try a clear leaf photo.") from exc
    if scores.size != len(CLASS_NAMES) or not np.isfinite(scores).all():
        raise DiseaseModelError("The CNN returned an invalid set of class scores.")
    if (scores < 0).any():
        raise DiseaseModelError("The CNN returned invalid negative class probabilities.")
    total = float(scores.sum())
    if total <= 0:
        raise DiseaseModelError("The CNN returned no usable class probabilities.")
    scores /= total
    ranking = np.argsort(scores)[::-1][: max(1, min(top_n, len(CLASS_NAMES)))]
    predictions: list[DiseasePrediction] = []
    for index in ranking:
        class_name = CLASS_NAMES[int(index)]
        crop_name, disease_name, display_name = _decode_label(class_name)
        predictions.append(DiseasePrediction(
            class_name=class_name,
            display_name=display_name,
            crop_name=crop_name,
            disease_name=disease_name,
            confidence=float(scores[index]),
        ))
    return predictions

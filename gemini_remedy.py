"""Gemini remedy suggestions with model fallback and no embedded credentials."""

from __future__ import annotations

from dataclasses import dataclass


PRIMARY_GEMINI_MODELS = {
    "Gemini 3.5 Flash-Lite": "gemini-3.5-flash-lite",
    "Gemini 3.8 Flash": "gemini-3.8-flash",
    "Gemini 3.6 Flash": "gemini-3.6-flash",
}
FALLBACK_GEMINI_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
)


class GeminiRemedyError(RuntimeError):
    """Raised when Gemini cannot return a remedy response."""


@dataclass(frozen=True)
class GeminiRemedy:
    text: str
    model_id: str


def generate_remedy(
    *,
    api_key: str,
    preferred_model_id: str,
    crop_name: str,
    disease_name: str,
    confidence: float,
    language: str,
) -> GeminiRemedy:
    """Ask Gemini for cautious integrated-pest-management guidance.

    The uploaded image is not sent to Gemini. Only the CNN label and confidence
    are included, so Gemini explains the classifier's result rather than
    independently diagnosing from the photo.
    """
    if not api_key.strip():
        raise GeminiRemedyError("Set GEMINI_API_KEY in Streamlit secrets or the environment first.")
    try:
        from google import genai
    except ImportError as exc:
        raise GeminiRemedyError(
            "The Gemini SDK is missing. Install the disease extras from requirements-disease.txt."
        ) from exc

    language_name = "Hindi" if language == "hi" else "English"
    certainty_note = (
        "The classifier confidence is low; clearly say that the diagnosis may be wrong and ask the user to confirm it."
        if confidence < 0.60 else
        "Treat the classifier result as a likely match, not a confirmed diagnosis."
    )
    prompt = f"""You are a cautious crop-care information assistant. Reply in {language_name}.

An image classifier predicted:
- Crop: {crop_name}
- Label: {disease_name}
- Classifier score: {confidence:.1%}

{certainty_note}
Give a short, practical response with these headings:
1. What the label means
2. Immediate low-risk steps (isolation if relevant, sanitation, water/airflow, monitoring)
3. Prevention and next checks
4. When to contact a local agricultural extension officer or plant clinic

Do not claim that the diagnosis is certain. Do not name pesticide brands, invent chemical doses,
or recommend tank mixes. If chemical treatment may be needed, tell the user to follow local
agricultural extension guidance and the locally registered product label. Distinguish general
guidance from crop- or region-specific advice. Keep the answer concise and understandable to a farmer."""

    model_ids = [preferred_model_id]
    model_ids.extend(model_id for model_id in FALLBACK_GEMINI_MODELS if model_id not in model_ids)
    client = genai.Client(api_key=api_key)
    failures: list[str] = []
    for model_id in model_ids:
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            answer = (getattr(response, "text", None) or "").strip()
            if not answer:
                failures.append(f"{model_id}: empty response")
                continue
            return GeminiRemedy(text=answer, model_id=model_id)
        except Exception as exc:
            status = getattr(exc, "code", getattr(exc, "status_code", None))
            failures.append(f"{model_id}: {type(exc).__name__}")
            # Authentication and permission errors do not improve by changing models.
            if status in {401, 403}:
                break
    tried = ", ".join(model_ids[: len(failures)])
    raise GeminiRemedyError(
        "Gemini could not generate advice with the configured model/fallback chain. "
        f"Tried: {tried}. Check the replacement API key, model access, and network, then retry."
    )

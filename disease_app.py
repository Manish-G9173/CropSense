"""English/Hindi Streamlit page for local PlantVillage CNN diagnosis."""

from __future__ import annotations

import hashlib
import html
import os

import streamlit as st

from disease_model import (
    HF_MODEL_ID,
    DiseaseModelError,
    load_plant_disease_model,
    predict_disease,
)
from gemini_remedy import (
    FALLBACK_GEMINI_MODELS,
    PRIMARY_GEMINI_MODELS,
    GeminiRemedyError,
    generate_remedy,
)
from plant_upload import PlantUploadError, decode_plant_upload


HI_CLASS_LABELS = {
    "Apple___Apple_scab": "सेब · स्कैब रोग",
    "Apple___Black_rot": "सेब · काला सड़न रोग",
    "Apple___Cedar_apple_rust": "सेब · रतुआ रोग",
    "Apple___Healthy": "सेब · स्वस्थ",
    "Blueberry___Healthy": "ब्लूबेरी · स्वस्थ",
    "Cherry___Powdery_mildew": "चेरी · पाउडरी मिल्ड्यू",
    "Cherry___Healthy": "चेरी · स्वस्थ",
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot": "मक्का · सर्कोस्पोरा पत्ती धब्बा",
    "Corn___Common_rust": "मक्का · सामान्य रतुआ",
    "Corn___Northern_Leaf_Blight": "मक्का · उत्तरी पत्ती झुलसा",
    "Corn___Healthy": "मक्का · स्वस्थ",
    "Grape___Black_rot": "अंगूर · काला सड़न रोग",
    "Grape___Esca_(Black_Measles)": "अंगूर · एस्का (ब्लैक मीज़ल्स)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "अंगूर · पत्ती झुलसा",
    "Grape___Healthy": "अंगूर · स्वस्थ",
    "Orange___Haunglongbing_(Citrus_greening)": "संतरा · सिट्रस ग्रीनिंग",
    "Peach___Bacterial_spot": "आड़ू · जीवाणु धब्बा",
    "Peach___Healthy": "आड़ू · स्वस्थ",
    "Pepper,_bell___Bacterial_spot": "शिमला मिर्च · जीवाणु धब्बा",
    "Pepper,_bell___Healthy": "शिमला मिर्च · स्वस्थ",
    "Potato___Early_blight": "आलू · अगेती झुलसा",
    "Potato___Late_blight": "आलू · पछेती झुलसा",
    "Potato___Healthy": "आलू · स्वस्थ",
    "Raspberry___Healthy": "रास्पबेरी · स्वस्थ",
    "Soybean___Healthy": "सोयाबीन · स्वस्थ",
    "Squash___Powdery_mildew": "स्क्वैश · पाउडरी मिल्ड्यू",
    "Strawberry___Leaf_scorch": "स्ट्रॉबेरी · पत्ती झुलसा",
    "Strawberry___Healthy": "स्ट्रॉबेरी · स्वस्थ",
    "Tomato___Bacterial_spot": "टमाटर · जीवाणु धब्बा",
    "Tomato___Early_blight": "टमाटर · अगेती झुलसा",
    "Tomato___Late_blight": "टमाटर · पछेती झुलसा",
    "Tomato___Leaf_Mold": "टमाटर · पत्ती फफूंद",
    "Tomato___Septoria_leaf_spot": "टमाटर · सेप्टोरिया पत्ती धब्बा",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "टमाटर · स्पाइडर माइट",
    "Tomato___Target_Spot": "टमाटर · टार्गेट स्पॉट",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "टमाटर · पीली पत्ती मोड़ विषाणु",
    "Tomato___Tomato_mosaic_virus": "टमाटर · मोज़ेक विषाणु",
    "Tomato___Healthy": "टमाटर · स्वस्थ",
}


COPY = {
    "en": {
        "eyebrow": "MODEL 03 · LEAF DISEASE CHECK",
        "title": "Check a leaf photo",
        "intro": "Upload a clear crop-leaf photo or a PDF with the photo on its first page. The PlantVillage CNN runs locally; Gemini can suggest cautious next steps after the label is predicted.",
        "model_card": "38 PlantVillage classes · MobileNetV2 CNN · local image inference",
        "upload": "Upload a leaf photo (JPG, PNG, WEBP) or PDF",
        "upload_help": "Maximum 20 MB. For PDFs, the first page is rendered and classified.",
        "pdf_note": "Only the first PDF page is used for the CNN check.",
        "analyze": "Analyze leaf →",
        "loading_model": "Loading the PlantVillage CNN from Hugging Face (first use downloads the model)…",
        "running_model": "Checking the leaf image…",
        "model_error": "Disease model could not run",
        "result_title": "CNN result",
        "confidence": "model score",
        "top_matches": "Other likely matches",
        "low_confidence": "The model is not confident. Try a close, well-lit photo of one leaf, or confirm with a plant clinic.",
        "healthy": "The top match is a healthy-leaf class. Check the other matches if visible damage remains.",
        "remedy_title": "Care guidance",
        "remedy_intro": "Gemini will receive only the predicted crop, label, and score—not your uploaded image.",
        "remedy_model": "Gemini model",
        "remedy_button": "Get care guidance →",
        "remedy_loading": "Asking Gemini for cautious crop-care guidance…",
        "remedy_used": "Response from {model}",
        "remedy_error": "Remedy request failed",
        "no_key": "Gemini is optional. Add a replacement key as GEMINI_API_KEY in .streamlit/secrets.toml to enable care guidance.",
        "key_example": "GEMINI_API_KEY = \"paste-your-new-key-here\"",
        "fallback_title": "Model fallback order",
        "fallback_copy": "If the chosen model is unavailable, the app tries other Gemini models automatically.",
        "disclaimer": "This classifier recognizes 38 PlantVillage classes across 14 crops. Those images were collected in controlled conditions, so field photos can be misclassified. Scores are not calibrated probabilities. Confirm the result locally before treatment; this is not a substitute for an agricultural expert.",
        "model_source": "Hugging Face model card",
        "plantvillage_source": "PlantVillage dataset",
        "unknown_crop": "If your crop is not one of the model’s 14 supported crops, the result may be unreliable.",
        "language": "English",
    },
    "hi": {
        "eyebrow": "मॉडल 03 · पत्ती की बीमारी जाँच",
        "title": "पत्ती की तस्वीर जाँचें",
        "intro": "फसल की साफ़ तस्वीर या पहले पन्ने पर तस्वीर वाला PDF अपलोड करें। PlantVillage CNN स्थानीय रूप से चलेगा; लेबल मिलने के बाद Gemini सावधानीपूर्वक अगले कदम सुझा सकता है।",
        "model_card": "PlantVillage की 38 श्रेणियाँ · MobileNetV2 CNN · तस्वीर की स्थानीय जाँच",
        "upload": "पत्ती की तस्वीर (JPG, PNG, WEBP) या PDF अपलोड करें",
        "upload_help": "अधिकतम 20 MB। PDF में पहले पन्ने को तस्वीर बनाकर जाँचा जाएगा।",
        "pdf_note": "CNN जाँच के लिए PDF का केवल पहला पन्ना उपयोग होगा।",
        "analyze": "पत्ती की जाँच करें →",
        "loading_model": "Hugging Face से PlantVillage CNN लोड हो रहा है (पहली बार मॉडल डाउनलोड होगा)…",
        "running_model": "पत्ती की तस्वीर जाँची जा रही है…",
        "model_error": "बीमारी मॉडल नहीं चल पाया",
        "result_title": "CNN का परिणाम",
        "confidence": "मॉडल स्कोर",
        "top_matches": "अन्य संभावित पहचान",
        "low_confidence": "मॉडल का भरोसा कम है। एक पत्ती की पास से, अच्छी रोशनी वाली तस्वीर लें या प्लांट क्लिनिक से पुष्टि करें।",
        "healthy": "सबसे संभावित पहचान स्वस्थ पत्ती की है। नुकसान दिखने पर अन्य संभावित पहचान भी देखें।",
        "remedy_title": "देखभाल के सुझाव",
        "remedy_intro": "Gemini को केवल फसल, अनुमानित बीमारी और स्कोर मिलेगा—आपकी अपलोड की गई तस्वीर नहीं।",
        "remedy_model": "Gemini मॉडल",
        "remedy_button": "देखभाल के सुझाव पाएँ →",
        "remedy_loading": "Gemini से सावधानीपूर्ण फसल देखभाल के सुझाव माँगे जा रहे हैं…",
        "remedy_used": "जवाब का मॉडल: {model}",
        "remedy_error": "सुझाव पाने में समस्या",
        "no_key": "Gemini वैकल्पिक है। सुझाव चालू करने के लिए नया API key .streamlit/secrets.toml में GEMINI_API_KEY नाम से रखें।",
        "key_example": "GEMINI_API_KEY = \"अपना-नया-key-यहाँ-रखें\"",
        "fallback_title": "वैकल्पिक मॉडल क्रम",
        "fallback_copy": "चुना हुआ मॉडल उपलब्ध न होने पर ऐप अपने-आप दूसरे Gemini मॉडल आज़माएगा।",
        "disclaimer": "यह मॉडल 14 फसलों की PlantVillage की 38 श्रेणियाँ पहचानता है। वे नियंत्रित परिस्थितियों की तस्वीरें हैं, इसलिए खेत की तस्वीर गलत पहचानी जा सकती है। स्कोर प्रमाणित संभावनाएँ नहीं हैं। उपचार से पहले स्थानीय विशेषज्ञ से पुष्टि करें; यह कृषि विशेषज्ञ का विकल्प नहीं है।",
        "model_source": "Hugging Face मॉडल जानकारी",
        "plantvillage_source": "PlantVillage डेटासेट",
        "unknown_crop": "यदि आपकी फसल मॉडल की 14 समर्थित फसलों में नहीं है, तो परिणाम भरोसेमंद नहीं हो सकता।",
        "language": "Hindi",
    },
}


@st.cache_resource(show_spinner=False)
def _cached_plant_model():
    return load_plant_disease_model()


def _get_gemini_key() -> str:
    secret_value = ""
    try:
        secret_value = str(st.secrets.get("GEMINI_API_KEY", ""))
    except Exception:
        pass
    return secret_value.strip() or os.getenv("GEMINI_API_KEY", "").strip()


def _localized_label(item: dict[str, object], language: str) -> str:
    if language == "hi":
        return HI_CLASS_LABELS.get(str(item["class_name"]), str(item["display_name"]))
    return str(item["display_name"])


def render_disease_dashboard(language: str) -> None:
    t = COPY[language]
    st.markdown(
        f"<section class='hero'><div class='hero-mark'>{t['eyebrow']}</div>"
        f"<p class='eyebrow'>CROPSENSE · PLANT HEALTH</p><h1>{t['title']}</h1>"
        f"<p class='hero-note'>{t['intro']}</p></section>",
        unsafe_allow_html=True,
    )
    model_col, model_link_col = st.columns([2, 1])
    model_col.caption(t["model_card"])
    model_link_col.markdown(
        f"[{t['model_source']} ↗](https://huggingface.co/{HF_MODEL_ID}) · "
        f"[{t['plantvillage_source']} ↗](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)",
    )

    uploaded = st.file_uploader(
        t["upload"],
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        help=t["upload_help"],
        key="disease_upload",
    )
    if uploaded is None:
        st.info(t["unknown_crop"], icon="🌿")
        st.caption(t["disclaimer"])
        return

    payload = uploaded.getvalue()
    try:
        image, upload_note = decode_plant_upload(uploaded.name, payload)
    except PlantUploadError as exc:
        st.error(str(exc))
        return

    file_hash = hashlib.sha256(payload).hexdigest()
    if st.session_state.get("disease_image_hash") != file_hash:
        st.session_state.pop("disease_prediction", None)
        st.session_state.pop("disease_remedy", None)
        st.session_state["disease_image_hash"] = file_hash

    preview_col, action_col = st.columns([1, 1], gap="large")
    with preview_col, st.container(border=True):
        st.image(image, width="stretch")
        if upload_note:
            st.caption(t["pdf_note"])
    with action_col, st.container(border=True):
        st.markdown(f"<div class='section-kicker'>{t['result_title']}</div>", unsafe_allow_html=True)
        st.markdown(t["upload_help"])
        if st.button(t["analyze"], type="primary", width="stretch", key="run_disease_analysis"):
            try:
                with st.spinner(t["loading_model"]):
                    cnn = _cached_plant_model()
                with st.spinner(t["running_model"]):
                    result = predict_disease(cnn, image, top_n=3)
                st.session_state["disease_prediction"] = [
                    {
                        "class_name": item.class_name,
                        "display_name": item.display_name,
                        "crop_name": item.crop_name,
                        "disease_name": item.disease_name,
                        "confidence": item.confidence,
                    }
                    for item in result
                ]
                st.session_state.pop("disease_remedy", None)
            except DiseaseModelError as exc:
                st.error(f"{t['model_error']}: {exc}")

    results = st.session_state.get("disease_prediction")
    if not results:
        st.caption(t["disclaimer"])
        return

    top = results[0]
    top_display = html.escape(_localized_label(top, language))
    st.markdown(
        f"<div class='panel result-panel' style='margin-top:20px'>"
        f"<div class='section-kicker'>{t['result_title']}</div>"
        f"<h2 style='color:var(--mint);overflow-wrap:anywhere'>{top_display}</h2>"
        f"<p class='crop-score'>{t['confidence']}: {top['confidence']:.1%}</p></div>",
        unsafe_allow_html=True,
    )
    if top["confidence"] < 0.60:
        st.warning(t["low_confidence"])
    if top["disease_name"].casefold() == "healthy":
        st.info(t["healthy"], icon="🌱")

    with st.expander(t["top_matches"], expanded=len(results) > 1):
        for item in results[1:]:
            label, score = st.columns([4, 1])
            label.write(_localized_label(item, language))
            score.write(f"{item['confidence']:.1%}")
            st.progress(float(item["confidence"]))

    st.markdown(f"### {t['remedy_title']}")
    st.caption(t["remedy_intro"])
    model_labels = list(PRIMARY_GEMINI_MODELS)
    selected_label = st.selectbox(
        t["remedy_model"], model_labels,
        index=0, key="disease_gemini_model",
    )
    with st.expander(t["fallback_title"]):
        st.caption(t["fallback_copy"])
        st.code(" → ".join([
            PRIMARY_GEMINI_MODELS[selected_label],
            *[model for model in FALLBACK_GEMINI_MODELS if model != PRIMARY_GEMINI_MODELS[selected_label]],
        ]), language="text")

    api_key = _get_gemini_key()
    if not api_key:
        st.info(t["no_key"])
        st.code(t["key_example"], language="toml")
    if st.button(t["remedy_button"], disabled=not bool(api_key), key="generate_disease_remedy"):
        try:
            with st.spinner(t["remedy_loading"]):
                remedy = generate_remedy(
                    api_key=api_key,
                    preferred_model_id=PRIMARY_GEMINI_MODELS[selected_label],
                    crop_name=top["crop_name"],
                    disease_name=top["disease_name"],
                    confidence=float(top["confidence"]),
                    language=language,
                )
            st.session_state["disease_remedy"] = {
                "text": remedy.text,
                "model_id": remedy.model_id,
                "image_hash": file_hash,
            }
        except GeminiRemedyError as exc:
            st.error(f"{t['remedy_error']}: {exc}")

    remedy_result = st.session_state.get("disease_remedy")
    if remedy_result and remedy_result.get("image_hash") == file_hash:
        st.markdown(f"**{t['remedy_used'].format(model=remedy_result['model_id'])}**")
        st.markdown(remedy_result["text"])
    st.caption(t["disclaimer"])

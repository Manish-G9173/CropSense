"""Streamlit UI for the separate crop-yield regression workflow."""

from __future__ import annotations

from pathlib import Path

import plotly.express as px
import pandas as pd
import streamlit as st

from yield_model import (
    NUMERIC_FEATURES,
    YieldDatasetError,
    find_yield_dataset,
    predict_yield,
    train_and_compare_yield_models,
)


@st.cache_resource(show_spinner=False)
def _cached_yield_bundle(path_text: str, modified_ns: int, byte_size: int, model_revision: int):
    del modified_ns, byte_size, model_revision
    return train_and_compare_yield_models(Path(path_text), n_splits=5, random_state=42)


COPY = {
    "en": {
        "eyebrow": "MODEL 02 · CROP YIELD REGRESSION",
        "title": "Estimate crop yield",
        "intro": "Explore historical yield patterns across crop, season, state, area and rainfall. Compare several regression models and estimate a yield for a matching field profile.",
        "dataset_missing": "Add the provided crop_yield.csv or crop_yield.csv.zip under the project’s data folder to enable yield estimates.",
        "dataset_path": "Expected path: data/crop_yield.csv (the ZIP is also supported).",
        "setup": "The supplied archive has been extracted to data/crop_yield.csv for this workspace. The dataset is ignored by Git.",
        "spinner": "Comparing yield regression models with forward-chaining cross-validation…",
        "issue": "Yield model setup issue",
        "rows": "Records",
        "features": "Crop types",
        "train_rows": "Train rows",
        "test_rows": "Holdout rows",
        "profile_title": "Build a field profile",
        "profile_copy": "The example starts from the latest matching historical record. Replace it with your field’s values.",
        "input_scale_note": "Keep numeric inputs on the same scale as the source CSV. That file does not document the measurement unit for Yield, so the result is shown on its recorded scale.",
        "crop": "Crop",
        "season": "Season",
        "state": "State",
        "year": "Crop year",
        "area": "Cultivated area",
        "rainfall": "Annual rainfall",
        "fertilizer": "Fertilizer used",
        "pesticide": "Pesticide used",
        "choose_model": "Regression model",
        "model_help": "Defaults to the model with the lowest average CV RMSE. You can choose any compared model.",
        "predict": "Estimate yield →",
        "prediction": "Estimated yield",
        "yield_units": "source-data scale",
        "prediction_model": "Model used",
        "prediction_note": "This is an estimate from historical records, not a guaranteed farm yield.",
        "compare_title": "Regression model comparison",
        "compare_copy": "Lower MAE and RMSE mean smaller prediction errors. CV is forward-chaining over earlier years; the latest years are kept as a separate holdout.",
        "model_details": "How these models were evaluated",
        "yield_skew_note": "Yield values are highly right-skewed in this file; a few very large records can dominate RMSE. Consider MAE alongside RMSE.",
        "chart_title": "Forward-CV RMSE by model · lower is better",
        "model_col": "Model",
        "cv_mae_col": "CV MAE · mean ± SD",
        "cv_rmse_col": "CV RMSE · mean ± SD",
        "test_mae_col": "Holdout MAE",
        "test_rmse_col": "Holdout RMSE",
        "evaluation_caption": "Training years: {train_start}–{train_end} · future holdout: {test_start}–{test_end}. Errors use the Yield scale recorded in the file.",
        "leakage_note": "Production is left out because it reveals the harvested output. Logistic Regression predicts categories, so it is not used to predict a numeric yield.",
        "data_note": "The dataset contains historical state-level records from 1997–2020. Estimates reflect these records and are not guaranteed field yields; use them as an exploratory estimate.",
        "rows_dropped": "Cleaning removed {rows:,} invalid or duplicate rows; {missing:,} feature values are imputed within each training fold.",
        "training_info": "All preprocessing is fit inside each fold. Unknown category values are handled without refitting on the holdout.",
    },
    "hi": {
        "eyebrow": "मॉडल 02 · फसल उपज रिग्रेशन",
        "title": "फसल उपज का अनुमान",
        "intro": "फसल, मौसम, राज्य, क्षेत्रफल और वर्षा के ऐतिहासिक संबंध देखें। कई रिग्रेशन मॉडल की तुलना करें और मिलते-जुलते खेत की जानकारी से उपज का अनुमान पाएँ।",
        "dataset_missing": "उपज का अनुमान चालू करने के लिए दी गई crop_yield.csv या crop_yield.csv.zip फ़ाइल प्रोजेक्ट के data फ़ोल्डर में रखें।",
        "dataset_path": "अपेक्षित जगह: data/crop_yield.csv (ZIP फ़ाइल भी चलेगी)।",
        "setup": "इस workspace के लिए दी गई ZIP फ़ाइल को data/crop_yield.csv में निकाला गया है। डेटासेट Git में शामिल नहीं है।",
        "spinner": "समय-क्रम वाले क्रॉस-वैलिडेशन से उपज रिग्रेशन मॉडल की तुलना हो रही है…",
        "issue": "उपज मॉडल सेटअप में समस्या",
        "rows": "उपयोग योग्य रिकॉर्ड",
        "features": "फसल विकल्प",
        "train_rows": "पुराने वर्षों के ट्रेनिंग रिकॉर्ड",
        "test_rows": "बाद के वर्षों के टेस्ट रिकॉर्ड",
        "profile_title": "खेत की जानकारी भरें",
        "profile_copy": "उदाहरण में मिलता-जुलता सबसे नया ऐतिहासिक रिकॉर्ड भरा है। इसे अपने खेत के मानों से बदलें।",
        "input_scale_note": "संख्यात्मक मान स्रोत CSV के समान पैमाने पर भरें। उस फ़ाइल में Yield की माप इकाई नहीं दी गई है, इसलिए नतीजा उसी दर्ज पैमाने पर दिखता है।",
        "crop": "फसल",
        "season": "मौसम",
        "state": "राज्य",
        "year": "फसल का वर्ष",
        "area": "खेती का क्षेत्रफल",
        "rainfall": "वार्षिक वर्षा",
        "fertilizer": "खाद का उपयोग",
        "pesticide": "कीटनाशक का उपयोग",
        "choose_model": "रिग्रेशन मॉडल",
        "model_help": "शुरुआत में औसत CV RMSE सबसे कम वाला मॉडल चुना है। चाहें तो कोई भी तुलना किया गया मॉडल चुनें।",
        "predict": "उपज का अनुमान पाएँ →",
        "prediction": "अनुमानित उपज",
        "yield_units": "स्रोत डेटा का पैमाना",
        "prediction_model": "इस्तेमाल किया मॉडल",
        "prediction_note": "यह ऐतिहासिक रिकॉर्ड पर आधारित अनुमान है, खेत की उपज की गारंटी नहीं।",
        "compare_title": "रिग्रेशन मॉडल की तुलना",
        "compare_copy": "कम MAE और RMSE का अर्थ है छोटी अनुमान त्रुटि। CV में पुराने वर्षों से आगे के वर्षों का अनुमान लगाया जाता है; सबसे नए वर्ष अलग टेस्ट के लिए रखे गए हैं।",
        "model_details": "मॉडल का मूल्यांकन कैसे हुआ",
        "yield_skew_note": "इस फ़ाइल में Yield मान बहुत असमान रूप से फैले हैं; कुछ बहुत बड़े रिकॉर्ड RMSE को काफ़ी प्रभावित कर सकते हैं। RMSE के साथ MAE भी देखें।",
        "chart_title": "Forward CV RMSE · कम बेहतर",
        "model_col": "मॉडल",
        "cv_mae_col": "CV MAE · औसत ± SD",
        "cv_rmse_col": "CV RMSE · औसत ± SD",
        "test_mae_col": "टेस्ट MAE",
        "test_rmse_col": "टेस्ट RMSE",
        "evaluation_caption": "ट्रेनिंग वर्ष: {train_start}–{train_end} · बाद के टेस्ट वर्ष: {test_start}–{test_end}। त्रुटियाँ फ़ाइल में दर्ज Yield पैमाने में हैं।",
        "leakage_note": "Production को इनपुट में नहीं लेते क्योंकि उससे वास्तविक उपज पता चलती है। Logistic Regression श्रेणियाँ बताता है, इसलिए संख्यात्मक उपज के लिए इसका उपयोग नहीं किया।",
        "data_note": "डेटासेट में 1997–2020 के ऐतिहासिक राज्य-स्तर के रिकॉर्ड हैं। अनुमान इन्हीं पैटर्न पर आधारित है, खेत की उपज की गारंटी नहीं; इसे शुरुआती अनुमान मानें।",
        "rows_dropped": "डेटा सफाई में {rows:,} अमान्य या डुप्लिकेट पंक्तियाँ हटाईं; {missing:,} फीचर मान हर ट्रेनिंग fold में भरे जाते हैं।",
        "training_info": "हर fold में preprocessing केवल उसी fold के ट्रेनिंग डेटा पर फिट होती है। अनदेखी श्रेणियाँ भी संभाली जाती हैं।",
    },
}


def render_yield_dashboard(language: str) -> None:
    t = COPY[language]
    root = Path(__file__).resolve().parent
    dataset_path = find_yield_dataset(root)

    st.markdown(
        f"<section class='hero'><div class='hero-mark'>{t['eyebrow']}</div>"
        f"<p class='eyebrow'>CROPSENSE · REGRESSION</p><h1>{t['title']}</h1>"
        f"<p class='hero-note'>{t['intro']}</p></section>",
        unsafe_allow_html=True,
    )
    if dataset_path is None:
        st.warning(t["dataset_missing"])
        st.caption(t["dataset_path"])
        st.stop()

    try:
        stat = dataset_path.stat()
        with st.spinner(t["spinner"]):
            bundle = _cached_yield_bundle(str(dataset_path), stat.st_mtime_ns, stat.st_size, model_revision=2)
    except YieldDatasetError as exc:
        st.error(f"{t['issue']}: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"{t['issue']}: {exc}")
        st.stop()

    frame = bundle.dataset.frame
    metric_cols = st.columns(4)
    metric_cols[0].metric(t["rows"], f"{len(frame):,}")
    metric_cols[1].metric(t["features"], f"{frame['Crop'].nunique():,}")
    metric_cols[2].metric(t["train_rows"], f"{bundle.train_size:,}")
    metric_cols[3].metric(t["test_rows"], f"{bundle.test_size:,}")
    if bundle.dataset.rows_dropped or bundle.dataset.feature_values_imputed:
        st.caption(t["rows_dropped"].format(
            rows=bundle.dataset.rows_dropped,
            missing=bundle.dataset.feature_values_imputed,
        ))

    st.markdown(f"### {t['profile_title']}")
    st.caption(t["profile_copy"])
    st.caption(t["input_scale_note"])
    prediction_key = f"yield_prediction_{language}"
    with st.form("yield_profile", border=True):
        input_cols = st.columns(4)
        crop_options = sorted(frame["Crop"].dropna().astype(str).unique())
        season_options = sorted(frame["Season"].dropna().astype(str).unique())
        state_options = sorted(frame["State"].dropna().astype(str).unique())
        years = frame["Crop_Year"].dropna().astype(int)
        crop_default = crop_options.index("Rice") if "Rice" in crop_options else 0
        season_default = season_options.index("Kharif") if "Kharif" in season_options else 0
        state_default = state_options.index("Punjab") if "Punjab" in state_options else 0
        values: dict[str, object] = {}
        values["Crop"] = input_cols[0].selectbox(t["crop"], crop_options, index=crop_default, key="yield_crop")
        values["Season"] = input_cols[1].selectbox(t["season"], season_options, index=season_default, key="yield_season")
        values["State"] = input_cols[2].selectbox(t["state"], state_options, index=state_default, key="yield_state")
        matching_profile = frame.loc[
            (frame["Crop"].astype(str) == str(values["Crop"]))
            & (frame["Season"].astype(str) == str(values["Season"]))
            & (frame["State"].astype(str) == str(values["State"]))
        ].sort_values("Crop_Year")
        values["Crop_Year"] = input_cols[3].number_input(
            t["year"], min_value=int(years.min()), max_value=int(years.max()) + 5,
            value=int(years.max()) + 1, step=1, key="yield_year",
        )

        numeric_cols = st.columns(4)
        if matching_profile.empty:
            matching_profile = frame.loc[frame["Crop"].astype(str) == str(values["Crop"])].sort_values("Crop_Year")
        sample_profile = matching_profile.iloc[-1] if not matching_profile.empty else frame.iloc[-1]
        numeric_defaults = {}
        for feature in NUMERIC_FEATURES[1:]:
            value = sample_profile[feature]
            numeric_defaults[feature] = float(value) if pd.notna(value) else float(frame[feature].median())
        labels = {
            "Area": t["area"], "Annual_Rainfall": t["rainfall"],
            "Fertilizer": t["fertilizer"], "Pesticide": t["pesticide"],
        }
        for index, feature in enumerate(NUMERIC_FEATURES[1:]):
            maximum = float(frame[feature].max())
            widget_max = max(maximum * 2.0, numeric_defaults[feature] * 2.0, 1.0)
            values[feature] = numeric_cols[index].number_input(
                labels[feature], min_value=0.0, max_value=widget_max,
                value=min(numeric_defaults[feature], widget_max),
                step=max(widget_max / 100.0, 0.01), format="%.2f", key=f"yield_{feature}",
            )

        model_names = bundle.comparison["Model"].tolist()
        selected_model = st.selectbox(
            t["choose_model"], model_names,
            index=model_names.index(bundle.best_model_name), key="yield_model_choice",
            help=t["model_help"],
        )
        submitted = st.form_submit_button(t["predict"], type="primary", width="stretch")

    if submitted:
        st.session_state[prediction_key] = predict_yield(bundle.models[selected_model], values)
        st.session_state[f"yield_prediction_model_{language}"] = selected_model
    prediction = st.session_state.get(prediction_key)
    if prediction is not None:
        used_model = st.session_state.get(f"yield_prediction_model_{language}", bundle.best_model_name)
        st.markdown(
            f"<div class='panel result-panel' style='margin-top:18px'>"
            f"<div class='section-kicker'>{t['prediction']}</div>"
            f"<div style='display:flex;align-items:baseline;flex-wrap:wrap;gap:12px;margin:8px 0'>"
            f"<span style='color:var(--mint);font:700 clamp(2.5rem,5vw,4rem) "
            f"\"Space Grotesk\",sans-serif'>{prediction:,.3f}</span>"
            f"<span style='color:var(--muted);font-size:.9rem'>{t['yield_units']}</span></div>"
            f"<p class='panel-copy' style='margin-bottom:0'>{t['prediction_model']}: {used_model}. "
            f"{t['prediction_note']}</p></div>",
            unsafe_allow_html=True,
        )
    st.markdown(f"### {t['compare_title']}")
    st.caption(t["compare_copy"])
    st.caption(t["yield_skew_note"])
    with st.expander(t["model_details"]):
        st.markdown(t["leakage_note"])
        st.caption(t["training_info"])
    st.caption(t["evaluation_caption"].format(
        train_start=min(bundle.train_years), train_end=max(bundle.train_years),
        test_start=min(bundle.test_years), test_end=max(bundle.test_years),
    ))
    comparison = bundle.comparison.copy()
    comparison["CV MAE display"] = comparison.apply(lambda row: f"{row['CV MAE']:,.3f} ± {row['CV MAE SD']:,.3f}", axis=1)
    comparison["CV RMSE display"] = comparison.apply(lambda row: f"{row['CV RMSE']:,.3f} ± {row['CV RMSE SD']:,.3f}", axis=1)
    table = comparison[["Model", "CV MAE display", "CV RMSE display", "Test MAE", "Test RMSE"]].rename(columns={
        "Model": t["model_col"],
        "CV MAE display": t["cv_mae_col"],
        "CV RMSE display": t["cv_rmse_col"],
        "Test MAE": t["test_mae_col"],
        "Test RMSE": t["test_rmse_col"],
    })
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        t["test_mae_col"]: st.column_config.NumberColumn(format="%.3f"),
        t["test_rmse_col"]: st.column_config.NumberColumn(format="%.3f"),
    })

    chart_data = bundle.comparison.sort_values("CV RMSE", ascending=True)
    fig = px.bar(
        chart_data,
        x="CV RMSE",
        y="Model",
        orientation="h",
        color="CV RMSE",
        text=chart_data["CV RMSE"].map(lambda value: f"{value:,.2f}"),
        color_continuous_scale=[[0, "#48e1aa"], [1, "#a66ce7"]],
        title=t["chart_title"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>RMSE: %{x:,.3f}<extra></extra>")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#e9deee"),
        margin=dict(l=12, r=45, t=60, b=24), height=390,
        xaxis_title="RMSE · dataset yield units", yaxis_title="", coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown(f"<p class='disclaimer'>{t['data_note']}</p>", unsafe_allow_html=True)

"""CropSense: crop recommendations from the local Kaggle dataset."""

from __future__ import annotations

import html
import math
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import (
    FEATURES,
    DatasetError,
    find_dataset,
    recommend,
    train_and_evaluate,
)
from translations import UI_TEXT


st.set_page_config(
    page_title="CropSense · Smart crop guidance",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#f7effb; --muted:#b5a6c2; --plum:#10051a; --panel:#1b0b26; --line:#55316b; --mint:#48e1aa; --violet:#cf83ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { color:var(--ink); background:radial-gradient(ellipse at 7% 1%,#471331 0%,transparent 31%),radial-gradient(ellipse at 92% 1%,#29105b 0%,transparent 33%),linear-gradient(145deg,#10051a 0%,#160722 50%,#0b0613 100%); }
[data-testid="stAppViewContainer"] { background:transparent; }
[data-testid="stHeader"] { background:rgba(16,5,26,.78); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#180921 0%,#100719 100%); border-right:1px solid #422453; }
[data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
.block-container { max-width:1460px; padding-top:1.7rem; padding-bottom:3rem; }
h1,h2,h3 { color:var(--ink); font-family:'Space Grotesk',sans-serif; letter-spacing:-.035em; }
p,li,label { color:#e6ddec; }
.hero { position:relative; overflow:hidden; border:1px solid #754398; border-radius:26px; padding:30px 36px 28px; margin-bottom:24px; background:linear-gradient(120deg,rgba(35,12,46,.94),rgba(19,8,31,.88)); box-shadow:0 18px 65px rgba(0,0,0,.22); }
.hero:after { content:''; position:absolute; width:400px; height:280px; top:-180px; right:-45px; border-radius:50%; background:rgba(72,225,170,.12); filter:blur(65px); pointer-events:none; }
.eyebrow { font-family:'DM Mono',monospace; color:var(--mint); font-size:.73rem; text-transform:uppercase; letter-spacing:.16em; margin:0 0 10px; }
.hero h1 { color:var(--mint); font-size:clamp(2.7rem,5vw,4.8rem); line-height:.98; margin:0; font-weight:700; }
.hero-sub { margin:12px 0 0; color:#f4eafa; font-size:1.12rem; font-weight:600; }
.hero-note { margin:18px 0 0; max-width:760px; line-height:1.6; color:#c5b5ce; }
.hero-mark { position:absolute; right:28px; top:26px; border:1px solid rgba(72,225,170,.55); color:var(--mint); background:rgba(72,225,170,.08); border-radius:999px; padding:9px 14px; font:500 .72rem 'DM Mono',monospace; letter-spacing:.08em; }
.section-kicker { color:var(--violet); font:500 .72rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.13em; margin-bottom:6px; }
.panel { border:1px solid #533469; border-radius:20px; background:linear-gradient(145deg,rgba(34,15,45,.93),rgba(25,11,35,.95)); padding:22px 23px; min-height:100%; }
.panel h3 { font-size:1.2rem; margin:0 0 3px; }
.panel-copy { color:var(--muted); font-size:.91rem; margin:.15rem 0 1rem; }
.result-panel { border-color:rgba(72,225,170,.55); background:radial-gradient(ellipse at 100% 0%,rgba(72,225,170,.1),transparent 47%),linear-gradient(145deg,#1b1229,#190c26); }
.crop-label { color:var(--muted); font:500 .72rem 'DM Mono',monospace; letter-spacing:.11em; text-transform:uppercase; margin:5px 0 2px; }
.crop-name { color:var(--mint); font:700 clamp(2rem,3vw,3.2rem) 'Space Grotesk',sans-serif; line-height:1.05; text-transform:capitalize; margin:0 0 3px; overflow-wrap:anywhere; }
.crop-score { color:#e6ddec; font:500 .9rem 'DM Mono',monospace; margin:0 0 20px; }
.recommend-row { display:flex; justify-content:space-between; gap:12px; margin:14px 0 5px; color:#e8ddec; font-size:.9rem; }
.recommend-row span:last-child { color:var(--mint); font-family:'DM Mono',monospace; }
.meter { height:7px; background:#392345; border-radius:99px; overflow:hidden; }
.meter > span { display:block; height:100%; border-radius:99px; background:linear-gradient(90deg,#a66ce7,#48e1aa); }
.metric-card { border:1px solid #4f3063; background:rgba(31,14,43,.82); border-radius:17px; padding:16px 18px; min-height:96px; }
.metric-label { color:var(--muted); font:500 .68rem 'DM Mono',monospace; letter-spacing:.1em; text-transform:uppercase; }
.metric-value { color:var(--ink); font:700 1.8rem 'Space Grotesk',sans-serif; margin-top:5px; }
.metric-foot { color:#95849f; font-size:.75rem; margin-top:1px; }
.disclaimer { color:#aa9ab4; font-size:.78rem; line-height:1.5; border-top:1px solid #392345; margin-top:18px; padding-top:13px; }
.stButton > button, .stFormSubmitButton > button { border-radius:11px; border:1px solid #5d3b70; background:#271236; color:#f7effb; transition:all .18s ease; }
.stButton > button:hover, .stFormSubmitButton > button:hover { border-color:var(--mint); color:var(--mint); }
.stFormSubmitButton > button[kind="primary"] { border:0; background:var(--mint); color:#071810; font-weight:700; }
.stFormSubmitButton > button[kind="primary"]:hover { background:#7bf0c3; color:#071810; }
[data-testid="stMetric"] { background:#1e0d2b; border:1px solid #4f3063; border-radius:16px; padding:13px 16px; }
[data-testid="stMetricLabel"] { color:#b5a6c2; }
[data-testid="stMetricValue"] { color:#f7effb; }
[data-testid="stPlotlyChart"] { border-radius:14px; overflow:hidden; }
[data-testid="stVerticalBlockBorderWrapper"] { border:1px solid #533469; border-radius:20px; background:linear-gradient(145deg,rgba(34,15,45,.93),rgba(25,11,35,.95)); padding:20px 22px; }
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { background:#160a20; border-color:#513365; }
input { color:#fff !important; }
hr { border-color:#3a2548; }
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #3a2548; }
.stTabs [data-baseweb="tab"] { color:#c1b2ca; }
.stTabs [aria-selected="true"] { color:var(--mint) !important; }
@media (max-width:760px) { .hero { padding:25px 22px; } .hero-mark { position:static; display:inline-block; margin-bottom:15px; } .panel { padding:18px; } }
</style>
""", unsafe_allow_html=True)


DEFAULT_INPUTS = {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 23.0,
    "humidity": 78.0,
    "ph": 6.5,
    "rainfall": 110.0,
}
SAMPLE_INPUTS = {
    "N": 78.0,
    "P": 38.0,
    "K": 40.0,
    "temperature": 24.5,
    "humidity": 72.0,
    "ph": 6.3,
    "rainfall": 125.0,
}
for _feature, _value in DEFAULT_INPUTS.items():
    st.session_state.setdefault(f"input_{_feature}", _value)


def apply_input_set(values: dict[str, float]) -> None:
    for feature, value in values.items():
        st.session_state[f"input_{feature}"] = value
    st.session_state.pop("prediction", None)


def make_plot_layout(fig: go.Figure, height: int = 370) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#e9deee", size=12),
        margin=dict(l=12, r=18, t=14, b=20),
        height=height,
        coloraxis_colorbar=dict(outlinewidth=0, tickfont=dict(color="#c8b7d0")),
    )
    fig.update_xaxes(gridcolor="rgba(169,137,184,.13)", zerolinecolor="rgba(169,137,184,.18)")
    fig.update_yaxes(gridcolor="rgba(169,137,184,.13)", zerolinecolor="rgba(169,137,184,.18)")
    return fig


def metric_card(label: str, value: str, foot: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-foot">{html.escape(foot)}</div></div>',
        unsafe_allow_html=True,
    )


def np_is_finite(value: float) -> bool:
    """Validate values entered by the user without accepting NaN or infinity."""
    return math.isfinite(value)


with st.sidebar:
    current_language = "hi" if st.session_state.get("language_display") == "हिन्दी" else "en"
    language_display = st.selectbox(
        UI_TEXT[current_language]["language_label"],
        ["English", "हिन्दी"],
        key="language_display",
    )
language = "hi" if language_display == "हिन्दी" else "en"
T = UI_TEXT[language]

with st.sidebar:
    app_section = st.radio(
        T["app_section"],
        ["crop", "yield", "disease"],
        format_func=lambda page: {
            "crop": T["mode_crop"],
            "yield": T["mode_yield"],
            "disease": T["mode_disease"],
        }[page],
        key="app_section",
    )
if app_section == "yield":
    from yield_app import render_yield_dashboard

    render_yield_dashboard(language)
    st.stop()
if app_section == "disease":
    from disease_app import render_disease_dashboard

    render_disease_dashboard(language)
    st.stop()


def localized_crop(crop: str) -> str:
    return T["crops"].get(str(crop).casefold(), str(crop).title())


st.markdown("""
<section class="hero">
  <div class="hero-mark">{hero_mark}</div>
  <p class="eyebrow">{hero_eyebrow}</p>
  <h1>CropSense</h1>
  <p class="hero-sub">{hero_subtitle}</p>
  <p class="hero-note">{hero_note}</p>
</section>
""".format(**T), unsafe_allow_html=True)


dataset_path = find_dataset(Path(__file__).resolve().parent)
bundle = None
load_error = None
if dataset_path is not None:
    @st.cache_resource(show_spinner=T["training_spinner"])
    def get_bundle(path_text: str, modified_ns: int, byte_size: int):
        del modified_ns, byte_size  # Included in the cache key to refresh after a CSV change.
        return train_and_evaluate(Path(path_text))

    try:
        stat = dataset_path.stat()
        bundle = get_bundle(str(dataset_path), stat.st_mtime_ns, stat.st_size)
    except DatasetError as exc:
        load_error = str(exc)
    except Exception as exc:  # Keep startup failures visible and actionable in the UI.
        load_error = f"The model could not be prepared: {exc}"


with st.sidebar:
    st.markdown(f'<p class="eyebrow">{T["sidebar_eyebrow"]}</p>', unsafe_allow_html=True)
    st.markdown(f"### {T['profile_title']}")
    st.caption(T["profile_caption"])
    with st.expander(T["soil_card_guide_title"]):
        st.markdown(T["soil_card_guide_copy"])
        st.markdown(f"[{T['soil_card_official_link']}](https://soilhealth.dac.gov.in/)")
    quick_a, quick_b = st.columns(2)
    quick_a.button(T["sample_button"], width="stretch", on_click=apply_input_set, args=(SAMPLE_INPUTS,))
    quick_b.button(T["reset_button"], width="stretch", on_click=apply_input_set, args=(DEFAULT_INPUTS,))

    with st.form("field_inputs", border=True):
        st.markdown(f"**{T['soil_nutrients']}**")
        col_a, col_b = st.columns(2)
        with col_a:
            n_value = st.number_input(T["nitrogen"], min_value=0.0, max_value=300.0, step=1.0, format="%.1f", key="input_N", help=T["nitrogen_help"])
            p_value = st.number_input(T["phosphorus"], min_value=0.0, max_value=300.0, step=1.0, format="%.1f", key="input_P", help=T["phosphorus_help"])
        with col_b:
            k_value = st.number_input(T["potassium"], min_value=0.0, max_value=300.0, step=1.0, format="%.1f", key="input_K", help=T["potassium_help"])
            ph_value = st.number_input(T["soil_ph"], min_value=0.0, max_value=14.0, step=0.1, format="%.1f", key="input_ph", help=T["soil_ph_help"])
        st.markdown(f"**{T['growing_conditions']}**")
        col_c, col_d = st.columns(2)
        with col_c:
            temp_value = st.number_input(T["temperature"], min_value=-20.0, max_value=60.0, step=0.5, format="%.1f", key="input_temperature")
            humidity_value = st.number_input(T["humidity"], min_value=0.0, max_value=100.0, step=1.0, format="%.1f", key="input_humidity")
        with col_d:
            rainfall_value = st.number_input(T["rainfall"], min_value=0.0, max_value=5000.0, step=5.0, format="%.1f", key="input_rainfall", help=T["rainfall_help"])
            st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button(T["submit"], type="primary", width="stretch", disabled=bundle is None)

    if submit and bundle is not None:
        values = {
            "N": n_value, "P": p_value, "K": k_value,
            "temperature": temp_value, "humidity": humidity_value,
            "ph": ph_value, "rainfall": rainfall_value,
        }
        try:
            for feature, value in values.items():
                if not np_is_finite(value):
                    raise ValueError(f"{T['features'][feature]} must be a finite number.")
            st.session_state["prediction"] = {"values": values, "recommendations": recommend(bundle.model, values)}
        except (ValueError, KeyError) as exc:
            st.error(f"{T['input_error']}: {exc}")


if load_error:
    st.error(f"{T['dataset_model_issue']}: {load_error}")
    st.stop()
elif bundle is None:
    st.warning(T["dataset_required"])
    st.markdown("""
<div class="panel" style="margin-bottom:20px">
 <div class="section-kicker">{dataset_required_kicker}</div>
 <h3>{dataset_heading}</h3>
 <p class="panel-copy">{dataset_description}</p>
</div>
""".format(**T), unsafe_allow_html=True)
    with st.expander(T["dataset_setup_expander"]):
        st.markdown(T["dataset_setup_text"])
        st.code("py -m pip install kaggle\nmkdir data\nkaggle datasets download -d atharvaingle/crop-recommendation-dataset -p data --unzip", language="powershell")
    st.link_button(T["open_dataset"], "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
    st.stop()


# KPI row is computed from the actual cleaned dataset and held-out test split.
data = bundle.dataset.frame
metric_cols = st.columns(4)
with metric_cols[0]:
    metric_card(T["held_accuracy"], f"{bundle.accuracy:.1%}", f"{bundle.test_size:,} {T['test_rows_fixed']}")
with metric_cols[1]:
    metric_card(T["crop_varieties"], f"{data['label'].nunique()}", T["distinct_labels"])
with metric_cols[2]:
    metric_card(T["training_records"], f"{bundle.train_size:,}", f"{T['usable_rows']} {len(data):,}")
with metric_cols[3]:
    metric_card(T["input_signals"], "07", T["soil_climate_inputs"])

if bundle.dataset.rows_dropped or bundle.dataset.numeric_values_imputed:
    st.caption(
        T["data_preparation"].format(
            rows=bundle.dataset.rows_dropped,
            values=bundle.dataset.numeric_values_imputed,
        )
    )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
input_col, result_col = st.columns([1.05, .95], gap="large")
with input_col, st.container(border=True):
    st.markdown(f'<div class="section-kicker">{T["field_profile"]}</div><h3>{T["input_panel_title"]}</h3><p class="panel-copy">{T["input_panel_copy"]}</p>', unsafe_allow_html=True)
    selected = st.session_state.get("prediction")
    if selected:
        values = selected["values"]
        st.markdown("<div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:15px'>" + "".join(
            f"<span style='border:1px solid #533469;border-radius:10px;padding:8px 10px;color:#ded0e5;font:500 .74rem DM Mono,monospace'>{html.escape(T['features'][f])} &nbsp; <b style='color:#48e1aa'>{values[f]:g}</b></span>"
            for f in FEATURES
        ) + "</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='disclaimer'>{T['profile_summary_note']}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='margin-top:18px;padding:18px;border:1px dashed #63447a;border-radius:14px;color:#b5a6c2;line-height:1.6'>{T['profile_waiting']}</div>", unsafe_allow_html=True)

with result_col, st.container(border=True):
    st.markdown(f'<div class="section-kicker">{T["model_ranking"]}</div><h3>{T["suggested_crops"]}</h3>', unsafe_allow_html=True)
    selected = st.session_state.get("prediction")
    if selected:
        recs = selected["recommendations"]
        top_crop, top_probability = recs[0]
        top_crop_display = html.escape(localized_crop(top_crop))
        st.markdown(f"<p class='crop-label'>{T['best_match']}</p><p class='crop-name'>{top_crop_display}</p><p class='crop-score'>{top_probability:.1%} {T['model_score']}</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='meter'><span style='width:{min(max(top_probability * 100, 0), 100):.2f}%'></span></div>", unsafe_allow_html=True)
        for rank, (crop, probability) in enumerate(recs, start=1):
            if rank == 1:
                continue
            crop_html = html.escape(localized_crop(crop))
            st.markdown(
                f"<div class='recommend-row'><span>{rank:02d} &nbsp; {crop_html}</span><span>{probability:.1%}</span></div>"
                f"<div class='meter'><span style='width:{min(max(probability * 100, 0), 100):.2f}%'></span></div>",
                unsafe_allow_html=True,
            )
        st.markdown(f"<p class='disclaimer'>{T['recommendation_note']}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='crop-label' style='margin-top:26px'>{T['prediction_waiting']}</p><p style='color:#b5a6c2;line-height:1.6'>{T['prediction_waiting_copy']}</p>", unsafe_allow_html=True)


st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
st.markdown(f'<p class="section-kicker">{T["model_transparency"]}</p><h2 style="margin-top:0">{T["explore_data"]}</h2>', unsafe_allow_html=True)
tab_model, tab_patterns = st.tabs([T["tab_performance"], T["tab_patterns"]])
with tab_model:
    info_col, matrix_col = st.columns([.8, 1.7], gap="large")
    with info_col, st.container(border=True):
        st.markdown(f'<div class="section-kicker">{T["held_evaluation"]}</div><h3>{T["performance_title"]}</h3><p class="panel-copy">{T["split_description"]}</p>', unsafe_allow_html=True)
        st.caption(T["rf_split_caption"].format(train=bundle.train_size, test=bundle.test_size))
        st.info(T["confusion_explanation"], icon="ℹ️")
    with matrix_col, st.container(border=True):
        st.markdown(f'<div class="section-kicker">{T["confusion_kicker"]}</div><h3>{T["confusion_title"]}</h3>', unsafe_allow_html=True)
        matrix = bundle.confusion
        labels = [localized_crop(str(label)) for label in bundle.classes]
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=[[0, "#21112c"], [.45, "#743d99"], [1, "#48e1aa"]],
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=f"{T['actual_crop']}: %{{y}}<br>{T['predicted_crop']}: %{{x}}<br>%{{z}}<extra></extra>",
            colorbar=dict(title=T["rows_colorbar"], tickfont=dict(color="#c8b7d0")),
        ))
        fig.update_layout(xaxis_title=T["predicted_crop"], yaxis_title=T["actual_crop"], height=650, margin=dict(l=80, r=35, t=10, b=115), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="#e9deee"))
        fig.update_xaxes(tickangle=-45, side="bottom", gridcolor="rgba(0,0,0,0)")
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with tab_patterns:
    dist_col, importance_col = st.columns(2, gap="large")
    with dist_col, st.container(border=True):
        st.markdown(f'<div class="section-kicker">{T["label_balance"]}</div><h3>{T["distribution_title"]}</h3><p class="panel-copy">{T["distribution_copy"]}</p>', unsafe_allow_html=True)
        distribution = data["label"].value_counts().rename_axis("crop").reset_index(name="records").sort_values("records", ascending=True)
        distribution["crop_display"] = distribution["crop"].map(localized_crop)
        fig = px.bar(distribution, x="records", y="crop_display", orientation="h", color="records", color_continuous_scale=[[0, "#70418e"], [1, "#48e1aa"]], hover_data={"records": True, "crop_display": True})
        fig.update_layout(xaxis_title=T["dataset_rows_axis"], yaxis_title="", coloraxis_showscale=False)
        make_plot_layout(fig, max(360, 24 * len(distribution)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with importance_col, st.container(border=True):
        st.markdown(f'<div class="section-kicker">{T["model_signals"]}</div><h3>{T["importance_title"]}</h3><p class="panel-copy">{T["importance_copy"]}</p>', unsafe_allow_html=True)
        importance_data = bundle.feature_importance.copy()
        importance_data["feature"] = importance_data["feature"].map(lambda name: T["features"].get(name, name))
        fig = px.bar(importance_data, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=[[0, "#70418e"], [1, "#48e1aa"]], text=importance_data["importance"].map(lambda value: f"{value:.1%}"))
        fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate=f"%{{y}}<br>{T['relative_importance_axis']}: %{{x:.1%}}<extra></extra>")
        fig.update_layout(xaxis_title=T["relative_importance_axis"], yaxis_title="", coloraxis_showscale=False, xaxis_tickformat=".0%", xaxis_range=[0, min(1, float(bundle.feature_importance["importance"].max()) * 1.25)])
        make_plot_layout(fig, 370)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(T["importance_caveat"])

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f'<div class="section-kicker">{T["correlation_kicker"]}</div><h3>{T["correlation_title"]}</h3><p class="panel-copy">{T["correlation_copy"]}</p>', unsafe_allow_html=True)
        correlation = data[FEATURES].corr(method="pearson")
        feature_labels = [T["features"][feature] for feature in FEATURES]
        correlation_values = correlation.round(2).values
        correlation_figure = go.Figure(data=go.Heatmap(
            z=correlation.values,
            x=feature_labels,
            y=feature_labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[[0, "#a66ce7"], [.5, "#21112c"], [1, "#48e1aa"]],
            text=correlation_values,
            texttemplate="%{text:.2f}",
            textfont={"size": 13},
            colorbar=dict(title=T["pearson_colorbar"], tickfont=dict(color="#c8b7d0")),
            hovertemplate=f"%{{y}} × %{{x}}<br>{'सहसंबंध' if language == 'hi' else 'Correlation'}: %{{z:.3f}}<extra></extra>",
        ))
        correlation_figure.update_layout(
            height=560,
            margin=dict(l=145, r=35, t=15, b=120),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#e9deee"),
            xaxis_title="",
            yaxis_title="",
        )
        correlation_figure.update_xaxes(tickangle=-35, side="bottom", gridcolor="rgba(0,0,0,0)")
        correlation_figure.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", scaleanchor="x", scaleratio=1)
        st.plotly_chart(correlation_figure, width="stretch", config={"displayModeBar": False})
        st.caption(T["correlation_note"])

st.markdown(f"<div style='height:22px'></div><p class='disclaimer'>{T['footer_disclaimer']}</p>", unsafe_allow_html=True)

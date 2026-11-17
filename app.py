"""
SONAR PREDICTION - Rock vs Mine classifier
Streamlit web app: user enters 60 sonar readings -> trained ML model -> ROCK or MINE + confidence.
Run locally:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

import model_utils as mu

st.set_page_config(
    page_title="Sonar Prediction | Rock vs Mine",
    page_icon="🌊",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; max-width: 1150px;}
.hero {
    background: linear-gradient(135deg, #062a3d 0%, #0b1626 55%, #0a2a1f 100%);
    border: 1px solid #16324a; border-radius: 20px;
    padding: 2rem 2.2rem; margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2.4rem; color: #e6f7ff; letter-spacing: .5px;}
.hero p {color: #a9c4d6; margin: .6rem 0 0 0; font-size: 1.05rem; line-height: 1.6;}
.badge {
    display: inline-block; background: #0c3b4a; color: #67e8f9;
    border: 1px solid #155e75; border-radius: 999px;
    padding: .15rem .8rem; font-size: .8rem; margin: 0 .4rem .8rem 0;
}
.result {
    border-radius: 18px; padding: 1.6rem; text-align: center;
    border: 1px solid; margin-bottom: .8rem;
}
.result.mine {background: #2a0d12; border-color: #ef4444;}
.result.rock {background: #0b2a22; border-color: #10b981;}
.result .big {font-size: 3rem; font-weight: 800; letter-spacing: 3px; margin: 0;}
.result.mine .big {color: #f87171;}
.result.rock .big {color: #34d399;}
.result .sub {color: #cbd5e1; margin: .3rem 0 0 0;}
.card {
    background: #0b1626; border: 1px solid #16324a; border-radius: 14px;
    padding: 1rem 1.2rem; height: 100%;
}
.card h4 {margin: 0 0 .4rem 0; color: #67e8f9;}
.card p {margin: 0; color: #b6c8d6; font-size: .95rem; line-height: 1.55;}
.step {
    background: #0b1626; border: 1px solid #16324a; border-radius: 14px;
    padding: .9rem; text-align: center; color: #e6edf5;
}
.step small {color: #8fa9bb;}
.arrow {text-align: center; font-size: 1.6rem; color: #67e8f9; padding-top: .8rem;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Cached training (runs once per server start, takes a few seconds)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models on the sonar dataset...")
def get_models():
    return mu.train_all()


models, metrics, confusions, test_df = get_models()
data_df = mu.load_data()
feature_cols = list(range(mu.N_FEATURES))
mean_rock = data_df[data_df[60] == "R"][feature_cols].mean().values
mean_mine = data_df[data_df[60] == "M"][feature_cols].mean().values

# ----------------------------------------------------------------------------
# Session-state helpers (callbacks must run before the widgets are drawn)
# ----------------------------------------------------------------------------
if "readings_text" not in st.session_state:
    st.session_state["readings_text"] = ""


def load_sample(code: str):
    pool = test_df[test_df["label"] == code]
    row = pool.sample(1).iloc[0]
    st.session_state["readings_text"] = ", ".join(
        f"{v:.4f}" for v in row[feature_cols].values
    )


def clear_text():
    st.session_state["readings_text"] = ""


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌊 Sonar Prediction")
    st.caption("Rock vs Mine classifier")
    model_name = st.selectbox(
        "Choose ML model",
        list(models.keys()),
        index=2,
        help="All three models are trained on the same 208-sample sonar dataset.",
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.write("208 samples · 60 features")
    st.write(
        f"🪨 Rocks: {(data_df[60] == 'R').sum()}  \n💣 Mines: {(data_df[60] == 'M').sum()}"
    )
    st.markdown("---")
    st.caption("Built with Python, scikit-learn & Streamlit.")

model = models[model_name]

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <span class="badge">Machine Learning</span><span class="badge">Binary Classification</span>
  <span class="badge">Streamlit</span>
  <h1>🌊 SONAR PREDICTION PROJECT</h1>
  <p>
    Sonar sends sound pulses underwater and listens to the echo. Metal mines and natural rocks
    reflect these signals differently across frequencies. This project trains machine-learning
    models on <b>208 real sonar returns</b> (60 frequency-band readings each) to automatically tell
    whether an underwater object is a <b>ROCK</b> or a <b>MINE</b> and shows how confident the model is.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

tab_predict, tab_perf, tab_about = st.tabs(
    ["🔮 Predict", "📊 Model Performance", "ℹ️ About the Project"]
)

# ----------------------------------------------------------------------------
# TAB 1 - PREDICT
# ----------------------------------------------------------------------------
with tab_predict:
    mode = st.radio(
        "How do you want to enter sonar readings?",
        ["Paste 60 values", "Enter manually", "Upload CSV (batch)"],
        horizontal=True,
    )

    values = None

    if mode == "Paste 60 values":
        st.caption(
            "Paste 60 comma-separated readings (each between 0 and 1), "
            "or load a random sample from the held-out test set."
        )
        b1, b2, b3, _ = st.columns([1, 1, 1, 3])
        b1.button("🪨 Load Rock sample", on_click=load_sample, args=("R",))
        b2.button("💣 Load Mine sample", on_click=load_sample, args=("M",))
        b3.button("🗑️ Clear", on_click=clear_text)
        text = st.text_area(
            "Sonar readings",
            key="readings_text",
            height=130,
            placeholder="0.0200, 0.0371, 0.0428, ... (60 values)",
        )
        if st.button("🔍 Predict", type="primary"):
            try:
                values = mu.parse_readings(text)
            except ValueError as e:
                st.error(str(e))

    elif mode == "Enter manually":
        st.caption("Fill in all 60 frequency-band readings (0.0 – 1.0).")
        manual = []
        with st.form("manual_form"):
            cols = st.columns(6)
            for i in range(mu.N_FEATURES):
                manual.append(
                    cols[i % 6].number_input(
                        f"F{i + 1}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.0001,
                        format="%.4f",
                        key=f"f{i}",
                    )
                )
            submitted = st.form_submit_button("🔍 Predict", type="primary")
        if submitted:
            values = np.array(manual)

    else:
        st.caption(
            "Upload a CSV with 60 numeric columns (no header). An optional 61st label column is ignored."
        )
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            try:
                batch = pd.read_csv(up, header=None)
                if batch.shape[1] >= 61:
                    batch = batch.iloc[:, :60]
                if batch.shape[1] != 60:
                    st.error(f"Expected 60 columns, found {batch.shape[1]}.")
                else:
                    batch = batch.apply(pd.to_numeric)
                    probs = model.predict_proba(batch.values)
                    classes = list(model.classes_)
                    idx = probs.argmax(axis=1)
                    out = pd.DataFrame(
                        {
                            "Row": np.arange(1, len(batch) + 1),
                            "Prediction": [mu.LABELS[classes[i]] for i in idx],
                            "Confidence": [f"{probs[k, i] * 100:.1f}%" for k, i in enumerate(idx)],
                        }
                    )
                    st.success(f"Predicted {len(out)} rows with {model_name}.")
                    st.dataframe(out, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Download predictions (CSV)",
                        out.to_csv(index=False).encode(),
                        "sonar_predictions.csv",
                        "text/csv",
                    )
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the file: {e}")

    # ---- single prediction result ----
    if values is not None:
        if values.min() < 0 or values.max() > 1:
            st.warning("Some readings are outside 0–1. The dataset is normalised, so results may be unreliable.")
        code, conf, prob_map = mu.predict(model, values)
        label = mu.LABELS[code]
        css = "mine" if code == "M" else "rock"
        icon = "💣" if code == "M" else "🪨"
        msg = (
            "⚠️ Potential underwater mine detected." if code == "M" else "Natural rock formation detected."
        )

        left, right = st.columns([1, 1.4])
        with left:
            st.markdown(
                f"""
<div class="result {css}">
  <p class="big">{icon} {label}</p>
  <p class="sub">{msg}</p>
  <p class="sub">Model: <b>{model_name}</b></p>
</div>""",
                unsafe_allow_html=True,
            )
            st.metric("Prediction confidence", f"{conf * 100:.1f}%")
            st.progress(min(max(conf, 0.0), 1.0))
            c1, c2 = st.columns(2)
            c1.metric("🪨 Rock probability", f"{prob_map.get('R', 0) * 100:.1f}%")
            c2.metric("💣 Mine probability", f"{prob_map.get('M', 0) * 100:.1f}%")
        with right:
            st.markdown("**Your signal vs. average Rock / Mine profile**")
            chart = pd.DataFrame(
                {
                    "Your reading": values,
                    "Avg Rock": mean_rock,
                    "Avg Mine": mean_mine,
                },
                index=np.arange(1, mu.N_FEATURES + 1),
            )
            chart.index.name = "Frequency band"
            st.line_chart(chart, height=300)

        if conf < 0.65:
            st.info("The model is not very sure about this one. Sonar data is noisy, so treat low-confidence results with care.")

# ----------------------------------------------------------------------------
# TAB 2 - PERFORMANCE
# ----------------------------------------------------------------------------
with tab_perf:
    st.subheader("How well do the models perform?")
    st.caption(
        "Test accuracy = 20% of the data that the models never saw during training (42 samples). "
        "5-fold CV accuracy is a more stable estimate because it uses every sample for testing once."
    )
    show = (metrics * 100).round(1).astype(str) + "%"
    st.dataframe(show, use_container_width=True)
    st.bar_chart(metrics[["Test accuracy", "5-fold CV accuracy"]] * 100, height=300)

    st.markdown(f"**Confusion matrix – {model_name}** (test set)")
    cm = confusions[model_name]
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual Rock", "Actual Mine"],
            columns=["Predicted Rock", "Predicted Mine"],
        ),
        use_container_width=True,
    )
    st.caption(
        "The dataset is small (208 samples), so exact accuracy varies with the train/test split. "
        "This is a learning/demo project, not a real-world mine detection system."
    )

# ----------------------------------------------------------------------------
# TAB 3 - ABOUT
# ----------------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.write(
        "**Sonar Prediction** is an end-to-end machine learning project. It starts from the classic "
        "UCI *Sonar, Mines vs. Rocks* dataset and ends with a web app anyone can use. "
        "A user enters the sonar readings, the trained model analyses the echo pattern, and the "
        "app reports whether the object is a **Rock** or a **Mine** together with a confidence score."
    )

    s1, a1, s2, a2, s3 = st.columns([3, 0.6, 3, 0.6, 3])
    s1.markdown('<div class="step"><b>Streamlit Frontend</b><br><small>User enters sonar readings</small></div>', unsafe_allow_html=True)
    a1.markdown('<div class="arrow">➜</div>', unsafe_allow_html=True)
    s2.markdown('<div class="step"><b>Trained ML Model</b><br><small>Logistic Regression / Random Forest / SVM</small></div>', unsafe_allow_html=True)
    a2.markdown('<div class="arrow">➜</div>', unsafe_allow_html=True)
    s3.markdown('<div class="step"><b>Prediction Result</b><br><small>ROCK or MINE + confidence</small></div>', unsafe_allow_html=True)

    st.write("")
    k1, k2, k3 = st.columns(3)
    k1.markdown(
        '<div class="card"><h4>📡 The data</h4><p>208 sonar returns, each with 60 numbers that '
        "show the signal energy in different frequency bands (0 to 1). Labels: R = Rock, M = Mine.</p></div>",
        unsafe_allow_html=True,
    )
    k2.markdown(
        '<div class="card"><h4>🧠 The models</h4><p>Logistic Regression (the original project model), '
        "Random Forest and SVM. Features are standardised and models are compared on unseen data.</p></div>",
        unsafe_allow_html=True,
    )
    k3.markdown(
        '<div class="card"><h4>🎯 The goal</h4><p>Help detect dangerous underwater mines early and show '
        "how machine learning can find patterns humans can't easily see in raw signals.</p></div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        "**Tech stack:** Python · pandas · NumPy · scikit-learn · Streamlit  \n"
        "**Dataset:** UCI Machine Learning Repository – Connectionist Bench (Sonar, Mines vs. Rocks)"
    )

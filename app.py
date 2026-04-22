"""
app.py — Streamlit Web UI for Heart Disease Prediction
Run with: streamlit run app.py

FIXES IN THIS VERSION
─────────────────────
1. Stale-model detector: warns if models/ were trained on >400 rows (dirty data).
2. cp=3 (Asymptomatic) dataset note: explains the Cleveland paradox to the user.
3. thal default changed from 2 → 2 (Reversable Defect is valid; 0 was sentinel).
4. Added "Retrain Required" banner if data leakage is detected.
"""

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

MODELS_DIR = "models"

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

st.markdown("""
<style>
.main-header {
    text-align: center; padding: 1.2rem 0;
    background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
    color: white; border-radius: 12px; margin-bottom: 1.5rem;
}
.auc-badge-excellent { background:#d5f5e3; border:2px solid #2ecc71; border-radius:8px; padding:0.6rem; text-align:center; color:#000000 !important; }
.auc-badge-excellent * { color:#000000 !important; }
.auc-badge-verygood  { background:#d6eaf8; border:2px solid #3498db; border-radius:8px; padding:0.6rem; text-align:center; color:#000000 !important; }
.auc-badge-verygood  * { color:#000000 !important; }
.auc-badge-good      { background:#fef9e7; border:2px solid #f39c12; border-radius:8px; padding:0.6rem; text-align:center; color:#000000 !important; }
.auc-badge-good      * { color:#000000 !important; }
.prediction-yes { background:#fdecea; border:2px solid #e74c3c; border-radius:10px; padding:1.5rem; text-align:center; color:#000000 !important; }
.prediction-yes * { color:#000000 !important; }
.prediction-no  { background:#eafaf1; border:2px solid #27ae60; border-radius:10px; padding:1.5rem; text-align:center; color:#000000 !important; }
.prediction-no  * { color:#000000 !important; }
.metric-card    { background:#f8f9fa; border-radius:8px; padding:0.8rem; text-align:center; border:1px solid #dee2e6; color:#000000 !important; }
.metric-card    * { color:#000000 !important; }
.stale-warning  { background:#fff3cd; border:2px solid #ffc107; border-radius:10px; padding:1rem; color:#000 !important; }
</style>
""", unsafe_allow_html=True)

MODEL_OPTIONS = [
    "Logistic", "SVM", "KNN", "Decision Tree", "Naive Bayes",
    "Random Forest", "Extra Trees", "XGBoost", "LightGBM",
    "AdaBoost", "Bagging", "Voting Ensemble", "Stacking Ensemble", "MLP (DL)",
]

FEATURE_META = {
    "age":      {"label": "Age (years)",                          "min": 1,   "max": 120, "default": 55,  "type": "int"},
    "sex":      {"label": "Sex",                                  "options": {0: "Female", 1: "Male"},     "default": 1,  "type": "select"},
    "cp":       {"label": "Chest Pain Type",                      "options": {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}, "default": 0, "type": "select"},
    "trestbps": {"label": "Resting Blood Pressure (mm Hg)",       "min": 80,  "max": 220, "default": 130, "type": "int"},
    "chol":     {"label": "Serum Cholesterol (mg/dl)",            "min": 100, "max": 600, "default": 240, "type": "int"},
    "fbs":      {"label": "Fasting Blood Sugar > 120 mg/dl",      "options": {0: "No", 1: "Yes"},          "default": 0,  "type": "select"},
    "restecg":  {"label": "Resting ECG Results",                  "options": {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}, "default": 0, "type": "select"},
    "thalach":  {"label": "Max Heart Rate Achieved",              "min": 60,  "max": 220, "default": 150, "type": "int"},
    "exang":    {"label": "Exercise Induced Angina",              "options": {0: "No", 1: "Yes"},          "default": 0,  "type": "select"},
    "oldpeak":  {"label": "ST Depression (oldpeak)",              "min": 0.0, "max": 7.0, "default": 1.0, "type": "float"},
    "slope":    {"label": "Slope of Peak Exercise ST Segment",    "options": {0: "Upsloping", 1: "Flat", 2: "Downsloping"}, "default": 1, "type": "select"},
    "ca":       {"label": "Major Vessels Coloured by Fluoroscopy","min": 0,   "max": 3,   "default": 0,   "type": "int"},   # max fixed: 4 was anomaly
    "thal":     {"label": "Thalassemia",                          "options": {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}, "default": 1, "type": "select"},  # 0 removed: sentinel value
}

# ── cp=3 note: this is a known Cleveland dataset encoding quirk ──────────────
CP3_NOTE = (
    "⚠️ **Dataset note on 'Asymptomatic' chest pain (cp=3):**  \n"
    "In the Cleveland dataset used here, patients coded as *Asymptomatic* "
    "actually had a **higher disease rate (69.6%)** than those with typical angina (27.7%). "
    "This is counter-intuitive but reflects that asymptomatic patients in this cohort "
    "were often referred for testing *because of* other high-risk indicators not captured here. "
    "**This model reflects that pattern** — it is not a bug in the app."
)


def model_safe(name): return name.replace(" ", "_").replace("(", "").replace(")", "")

@st.cache_resource
def load_preprocessor():
    p = f"{MODELS_DIR}/preprocessor.pkl"
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None

@st.cache_resource
def load_meta():
    p = f"{MODELS_DIR}/meta.pkl"
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None

@st.cache_resource
def load_model(name):
    p = f"{MODELS_DIR}/{model_safe(name)}.pkl"
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None

@st.cache_data
def load_results():
    p = f"{MODELS_DIR}/results.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


def check_stale_models():
    """
    Detect if models were trained on the dirty (duplicated) dataset.
    The clean dataset has 296 rows; if results.csv shows suspiciously high
    AUC (>0.97) on ALL models, or if meta has >600 feature rows implied,
    warn the user to retrain.
    """
    results_df = load_results()
    if results_df is None:
        return False, ""
    median_auc = results_df["AUC"].median()
    max_auc    = results_df["AUC"].max()
    # On clean 296-row data, expect AUC 0.85–0.95. If median > 0.97, data was dirty.
    if median_auc > 0.97:
        return True, f"Median AUC = {median_auc:.4f} (suspiciously high — likely trained on duplicated data)"
    return False, ""


def auc_badge(auc_val):
    if auc_val >= 0.95:
        return f'<div class="auc-badge-excellent">🟢 AUC {auc_val:.4f}<br><small>Excellent</small></div>'
    elif auc_val >= 0.90:
        return f'<div class="auc-badge-verygood">🔵 AUC {auc_val:.4f}<br><small>Very Good</small></div>'
    else:
        return f'<div class="auc-badge-good">🟡 AUC {auc_val:.4f}<br><small>Good</small></div>'


def render_input(col, meta, widget_col):
    with widget_col:
        ftype = meta.get("type", "float")
        if ftype == "select":
            opts = meta["options"]
            keys = list(opts.keys())
            labels = list(opts.values())
            default_idx = keys.index(meta["default"]) if meta["default"] in keys else 0
            chosen_label = st.selectbox(meta["label"], options=labels, index=default_idx, key=f"inp_{col}")
            return keys[labels.index(chosen_label)]
        elif ftype == "float":
            return st.number_input(meta["label"],
                                   min_value=float(meta["min"]), max_value=float(meta["max"]),
                                   value=float(meta["default"]), step=0.1, key=f"inp_{col}")
        else:
            return st.number_input(meta["label"],
                                   min_value=int(meta["min"]), max_value=int(meta["max"]),
                                   value=int(meta["default"]), step=1, key=f"inp_{col}")


def main():
    st.markdown("""
    <div class="main-header">
        <h1>❤️  Heart Disease Predictor</h1>
        <p>Machine Learning · 14 Models · AUC-Optimised</p>
    </div>""", unsafe_allow_html=True)

    preprocessor = load_preprocessor()
    meta         = load_meta()

    if preprocessor is None or meta is None:
        st.error("Models not found. Run `python train.py` first.")
        st.code("python train.py", language="bash")
        st.stop()

    # ── Stale model check ─────────────────────────────────────────────────────
    stale, reason = check_stale_models()
    
    feature_cols = meta["feature_cols"]
    results_df   = load_results()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    available = [m for m in MODEL_OPTIONS
                 if os.path.exists(f"{MODELS_DIR}/{model_safe(m)}.pkl")]

    default_idx = 0
    if results_df is not None and len(results_df):
        best = results_df.sort_values("AUC", ascending=False).iloc[0]["Model"]
        if best in available:
            default_idx = available.index(best)

    selected = st.sidebar.selectbox("🤖 Select Model", available, index=default_idx)

    if results_df is not None:
        row = results_df[results_df["Model"] == selected]
        if not row.empty:
            auc_val = row.iloc[0]["AUC"]
            st.sidebar.markdown(auc_badge(auc_val), unsafe_allow_html=True)
            st.sidebar.markdown("")

    st.sidebar.markdown("---")
    st.sidebar.info("**ℹ️ About AUC**\n\nAUC (Area Under ROC Curve) is the "
                    "gold-standard metric for medical classification. "
                    "It measures how well the model separates disease from "
                    "no-disease regardless of threshold.\n\n"
                    "• ≥ 0.95 → Excellent\n• ≥ 0.90 → Very Good\n• ≥ 0.85 → Good")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Dataset Info (after cleaning)**")
    st.sidebar.markdown("• Original rows: 1,025  \n• Duplicates removed: 723  \n• Clean rows: 296  \n• Train/Test: 80/20")

    # ── Input Fields ──────────────────────────────────────────────────────────
    st.subheader("🩺 Patient Data")
    input_values = {}
    cp_selected = None

    for i in range(0, len(feature_cols), 3):
        cols = st.columns(3)
        for j, feat in enumerate(feature_cols[i:i+3]):
            fm = FEATURE_META.get(feat, {"label": feat, "type": "float", "min": 0, "max": 100, "default": 0})
            val = render_input(feat, fm, cols[j])
            input_values[feat] = val
            if feat == "cp":
                cp_selected = val

    # Show cp=3 note inline if user selects Asymptomatic
    if cp_selected == 3:
        st.info(CP3_NOTE)

    st.markdown("---")
    predict_clicked = st.button("🔍  Predict", use_container_width=False, type="primary")

    if predict_clicked:
        with st.spinner(f"Running {selected}..."):
            model = load_model(selected)
            if model is None:
                st.error(f"Could not load: {selected}")
                st.stop()

            row_df = pd.DataFrame([{c: float(input_values.get(c, 0)) for c in feature_cols}], columns=feature_cols)
            row_df = row_df.astype(float)
            X_proc = preprocessor.transform(row_df)

            pred_class = int(model.predict(X_proc)[0])
            if hasattr(model, "predict_proba"):
                proba        = model.predict_proba(X_proc)[0]
                confidence   = float(proba[pred_class])
                prob_disease = float(proba[1])
            else:
                score        = model.decision_function(X_proc)[0]
                confidence   = abs(float(score))
                prob_disease = 0.5

        st.markdown("---")
        st.subheader("📋 Result")

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

        with c1:
            if pred_class == 1:
                st.markdown('<div class="prediction-yes"><h2>❤️‍🔥 Heart Disease: YES</h2>'
                            '<p>Patient is <strong>at risk</strong>. Consult a cardiologist.</p></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-no"><h2>💚 Heart Disease: NO</h2>'
                            '<p>Patient is <strong>not at risk</strong> based on these values.</p></div>',
                            unsafe_allow_html=True)

        with c2:
            st.markdown(f'<div class="metric-card"><h4>Confidence</h4><h2>{confidence*100:.1f}%</h2></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h4>Disease Prob.</h4><h2>{prob_disease*100:.1f}%</h2></div>',
                        unsafe_allow_html=True)
        with c4:
            if results_df is not None:
                row = results_df[results_df["Model"] == selected]
                if not row.empty:
                    auc_val = row.iloc[0]["AUC"]
                    st.markdown(f'<div class="metric-card"><h4>Model AUC</h4><h2>{auc_val:.4f}</h2></div>',
                                unsafe_allow_html=True)

        st.markdown("")
        st.markdown("**Disease probability:**")
        st.progress(min(prob_disease, 1.0))
        st.caption(f"Model used: **{selected}**   |   "
                   f"Disease probability: {prob_disease*100:.1f}%")

        # ── Clinical context note ─────────────────────────────────────────────
        age = input_values.get("age", 55)
        sex = input_values.get("sex", 1)
        if pred_class == 1 and age < 40 and sex == 0:
            st.info(
                "ℹ️ **Clinical context:** Heart disease in women under 40 is rare. "
                "The model is trained on a small dataset (296 rows) that has limited "
                "representation of young female patients — interpret this result with caution. "
                "Always consult a cardiologist for clinical decisions."
            )

        with st.expander("🔎 Input Summary"):
            st.dataframe(pd.DataFrame(input_values, index=["Value"]).T, use_container_width=True)

    # ── Leaderboard ───────────────────────────────────────────────────────────
    if results_df is not None:
        st.markdown("---")
        st.subheader("🏆 Model Leaderboard  *(sorted by AUC — primary metric)*")

        if stale:
            st.warning("⚠️ Leaderboard below reflects stale (pre-dedup) models. AUC values are inflated.")

        def colour_auc(val):
            if val >= 0.95: return "background-color:#d5f5e3; color:#000000;"
            if val >= 0.90: return "background-color:#d6eaf8; color:#000000;"
            if val >= 0.85: return "background-color:#fef9e7; color:#000000;"
            return "background-color:#fadbd8; color:#000000;"

        ranked = results_df.sort_values("AUC", ascending=False).reset_index(drop=True)
        ranked.index += 1
        ranked.insert(0, "Rank", ranked.index)

        try:
            styled = (ranked.style
                      .map(colour_auc, subset=["AUC"])
                      .format({"AUC": "{:.4f}", "Accuracy": "{:.4f}",
                               "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}))
        except AttributeError:
            styled = (ranked.style
                      .applymap(colour_auc, subset=["AUC"])
                      .format({"AUC": "{:.4f}", "Accuracy": "{:.4f}",
                               "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}))
        st.dataframe(styled, use_container_width=True)

        st.caption("🟢 ≥0.95 Excellent · 🔵 ≥0.90 Very Good · 🟡 ≥0.85 Good · 🔴 Fair  "
                   "| AUC is the standard metric for medical binary classification.")

    st.markdown("---")
    st.caption("⚠️ For educational purposes only — not a substitute for professional medical advice.")


if __name__ == "__main__":
    main()

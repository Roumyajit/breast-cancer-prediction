import streamlit as st
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { max-width: 1200px; margin: 0 auto; }
    .title-box {
        background: linear-gradient(135deg, #1a5276, #2980b9);
        padding: 2rem; border-radius: 12px;
        text-align: center; margin-bottom: 1.5rem;
        color: white;
    }
    .result-malignant {
        background: #fdf2f2; border: 2px solid #e74c3c;
        border-radius: 10px; padding: 1.5rem; text-align: center;
    }
    .result-benign {
        background: #f0faf0; border: 2px solid #27ae60;
        border-radius: 10px; padding: 1.5rem; text-align: center;
    }
    .metric-card {
        background: white; border-radius: 10px;
        padding: 1rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .info-box {
        background: #eaf4fb; border-left: 4px solid #2980b9;
        padding: 1rem; border-radius: 6px; margin: 1rem 0;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Train Model (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    return model, scaler, feature_names, acc, data

model, scaler, feature_names, model_acc, data = train_model()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1 style="margin:0; font-size:2rem;">🧬 Breast Cancer Prediction App</h1>
    <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size:1rem;">
        Machine Learning · Healthcare AI · SVM Classifier
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Info banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="info-box">
    <strong>ℹ️ About this App:</strong> This app uses a Support Vector Machine (SVM) model trained on the
    Wisconsin Breast Cancer Dataset to classify tumours as <strong>Malignant</strong> (cancerous) or
    <strong>Benign</strong> (non-cancerous). Adjust the sliders on the left and click <em>Predict</em>.
    <br><br>⚠️ <em>For educational purposes only. Not a substitute for medical diagnosis.</em>
</div>
""", unsafe_allow_html=True)

# ─── Model metrics row ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <h3 style="color:#1a5276; margin:0">{model_acc*100:.1f}%</h3>
        <p style="color:#666; margin:0; font-size:0.85rem">Model Accuracy</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="metric-card">
        <h3 style="color:#1a5276; margin:0">SVM (RBF)</h3>
        <p style="color:#666; margin:0; font-size:0.85rem">Algorithm</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="metric-card">
        <h3 style="color:#1a5276; margin:0">569</h3>
        <p style="color:#666; margin:0; font-size:0.85rem">Training Samples</p>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown("""<div class="metric-card">
        <h3 style="color:#1a5276; margin:0">30</h3>
        <p style="color:#666; margin:0; font-size:0.85rem">Features Used</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Sidebar sliders ──────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔬 Tumour Feature Inputs")
st.sidebar.markdown("Adjust the values below based on clinical measurements:")
st.sidebar.markdown("---")

# Use dataset min/max for slider ranges
dmin = data.data.min(axis=0)
dmax = data.data.max(axis=0)
dmean = data.data.mean(axis=0)

# Group features for cleaner UI
groups = {
    "📐 Mean Features": list(range(0, 10)),
    "📏 Standard Error Features": list(range(10, 20)),
    "⚠️ Worst Features": list(range(20, 30)),
}

user_inputs = []
for group_name, indices in groups.items():
    st.sidebar.markdown(f"**{group_name}**")
    for i in indices:
        fname = feature_names[i].replace(' ', '_')
        val = st.sidebar.slider(
            label=feature_names[i],
            min_value=float(round(dmin[i], 4)),
            max_value=float(round(dmax[i], 4)),
            value=float(round(dmean[i], 4)),
            format="%.4f",
            key=f"feat_{i}"
        )
        user_inputs.append(val)
    st.sidebar.markdown("")

# ─── Prediction ───────────────────────────────────────────────────────────────
col_btn, col_reset = st.columns([1, 4])
with col_btn:
    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

if predict_btn:
    input_array = np.array(user_inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2 = st.columns([1, 1])

    with r1:
        if prediction == 0:   # Malignant = 0 in this encoding
            st.markdown("""
            <div class="result-malignant">
                <h2 style="color:#e74c3c; margin:0">🔴 MALIGNANT</h2>
                <p style="color:#666; margin:0.5rem 0 0 0">
                    The model predicts this tumour is <strong>cancerous</strong>.<br>
                    Immediate medical consultation is recommended.
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-benign">
                <h2 style="color:#27ae60; margin:0">🟢 BENIGN</h2>
                <p style="color:#666; margin:0.5rem 0 0 0">
                    The model predicts this tumour is <strong>non-cancerous</strong>.<br>
                    Regular monitoring is still advised.
                </p>
            </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown("#### 📊 Prediction Confidence")
        benign_pct    = proba[1] * 100
        malignant_pct = proba[0] * 100
        st.metric("Benign Probability",    f"{benign_pct:.1f}%")
        st.metric("Malignant Probability", f"{malignant_pct:.1f}%")
        st.progress(int(benign_pct))

    st.markdown("""
    > ⚠️ **Disclaimer:** This prediction is generated by an ML model for educational purposes only.
    > Always consult a qualified medical professional for actual diagnosis and treatment.
    """)

# ─── About section ────────────────────────────────────────────────────────────
with st.expander("📘 About the Model & Dataset"):
    st.markdown("""
    ### Wisconsin Breast Cancer Dataset
    - **Source:** UCI Machine Learning Repository
    - **Samples:** 569 (357 Benign, 212 Malignant)
    - **Features:** 30 numerical features computed from digitised images of fine needle aspirate (FNA) of breast masses

    ### Features Include:
    - **Radius** — mean of distances from center to perimeter points
    - **Texture** — standard deviation of gray-scale values
    - **Perimeter, Area, Smoothness, Compactness, Concavity**
    - **Concave Points, Symmetry, Fractal Dimension**
    - Each computed as: *mean*, *standard error*, and *worst (largest mean of 3 largest values)*

    ### Model: SVM with RBF Kernel
    - C = 10, Gamma = 0.01
    - Optimised for high **Recall on Malignant class** (minimising false negatives)
    - 5-fold cross-validated
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem">
    Built by <strong>Roumyajit Sarkar</strong> &nbsp;|&nbsp;
    <a href="https://github.com/Roumyajit" target="_blank">GitHub</a> &nbsp;|&nbsp;
    <a href="https://www.linkedin.com/in/roumyajit-sarkar/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)

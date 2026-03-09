import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import io
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Occult DKD Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Space+Mono:wght@400;700&display=swap');

/* Root Variables */
:root {
    --deep-navy: #0d1b2a;
    --slate-blue: #1b2f44;
    --accent-teal: #00c9b1;
    --accent-amber: #f4a535;
    --accent-coral: #e85d75;
    --soft-white: #f0eeea;
    --muted: #8899aa;
    --card-bg: #162535;
}

/* Global Reset */
.stApp {
    background-color: var(--deep-navy);
    font-family: 'Crimson Pro', Georgia, serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1520 0%, #0d1b2a 40%, #0a1218 100%);
    border-right: 1px solid rgba(0, 201, 177, 0.15);
}

[data-testid="stSidebar"] * {
    color: var(--soft-white) !important;
}

/* Main content area */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1200px;
}

/* Hero Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(0, 201, 177, 0.2);
    margin-bottom: 2rem;
}

.hero-title {
    font-family: 'Crimson Pro', serif;
    font-size: 3rem;
    font-weight: 300;
    color: var(--soft-white);
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.1;
}

.hero-title em {
    font-style: italic;
    color: var(--accent-teal);
}

.hero-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.75rem;
}

/* Section Labels */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent-teal);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(0, 201, 177, 0.2);
}

/* Input Cards */
.input-card {
    background: var(--card-bg);
    border: 1px solid rgba(0, 201, 177, 0.12);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}

.input-card:hover {
    border-color: rgba(0, 201, 177, 0.3);
}

/* Streamlit Inputs — Labels */
.stNumberInput label, .stSelectbox label, .stSlider label {
    font-family: 'Crimson Pro', serif !important;
    font-size: 1rem !important;
    color: var(--soft-white) !important;
    font-weight: 400 !important;
}

/* Number Input — kill white backgrounds at every level */
.stNumberInput > div,
.stNumberInput > div > div,
.stNumberInput > div > div > div {
    background-color: #162535 !important;
    background: #162535 !important;
}

/* The actual <input> element */
.stNumberInput input {
    background-color: #162535 !important;
    background: #162535 !important;
    border: 1px solid rgba(0, 201, 177, 0.25) !important;
    border-radius: 8px !important;
    color: var(--soft-white) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stNumberInput input:focus {
    border-color: var(--accent-teal) !important;
    box-shadow: 0 0 0 2px rgba(0, 201, 177, 0.15) !important;
    outline: none !important;
}

/* Number input +/- buttons */
.stNumberInput button {
    background-color: #1e3550 !important;
    border: 1px solid rgba(0, 201, 177, 0.2) !important;
    color: var(--accent-teal) !important;
}

.stNumberInput button:hover {
    background-color: #243f63 !important;
    border-color: var(--accent-teal) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] > div > div > div {
    background-color: #162535 !important;
    border: 1px solid rgba(0, 201, 177, 0.25) !important;
    border-radius: 8px !important;
    color: var(--soft-white) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-teal) 0%, #00a896 100%) !important;
    color: var(--deep-navy) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #00e5ca 0%, #00c9b1 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0, 201, 177, 0.3) !important;
}

/* Result Box */
.result-high {
    background: linear-gradient(135deg, rgba(232, 93, 117, 0.15) 0%, rgba(232, 93, 117, 0.05) 100%);
    border: 1px solid rgba(232, 93, 117, 0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-low {
    background: linear-gradient(135deg, rgba(0, 201, 177, 0.15) 0%, rgba(0, 201, 177, 0.05) 100%);
    border: 1px solid rgba(0, 201, 177, 0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-value {
    font-family: 'Crimson Pro', serif;
    font-size: 4rem;
    font-weight: 300;
    line-height: 1;
    margin: 0;
}

.result-interpretation {
    font-family: 'Crimson Pro', serif;
    font-size: 1.15rem;
    color: var(--soft-white);
    margin-top: 1rem;
    line-height: 1.6;
    font-style: italic;
}

/* Feature importance bars */
.feature-bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--soft-white);
}

/* Info boxes */
.info-box {
    background: rgba(244, 165, 53, 0.08);
    border-left: 3px solid var(--accent-amber);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}

.info-box p {
    font-family: 'Crimson Pro', serif;
    font-size: 0.95rem;
    color: var(--soft-white);
    margin: 0;
    line-height: 1.6;
}

/* Dividers */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 1.5rem 0;
}

/* Metric tags */
.metric-tag {
    display: inline-block;
    background: rgba(0, 201, 177, 0.1);
    border: 1px solid rgba(0, 201, 177, 0.25);
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent-teal);
    letter-spacing: 0.05em;
    margin: 0.15rem;
}

/* Streamlit default overrides */
.stMarkdown p {
    color: var(--soft-white);
    font-family: 'Crimson Pro', serif;
    font-size: 1rem;
}

/* Slider */
.stSlider .st-bd { background: rgba(0, 201, 177, 0.2) !important; }
.stSlider .st-be { background: var(--accent-teal) !important; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--card-bg) !important;
    border: 1px solid rgba(0, 201, 177, 0.12) !important;
    border-radius: 8px !important;
    color: var(--soft-white) !important;
    font-family: 'Crimson Pro', serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(0, 201, 177, 0.15);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.1em;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-teal) !important;
    border-bottom: 2px solid var(--accent-teal) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '8特征模型-LR.pkl')
    return joblib.load(model_path)

model = load_model()

FEATURES = ['HGB', 'HbA1c', 'HTN', 'UA', 'sex', 'MicroVCs', 'CVD', 'A/G']
COEF = model.coef_[0]
INTERCEPT = model.intercept_[0]

# Feature descriptions
FEATURE_INFO = {
    'HGB':      {'label': 'Hemoglobin (HGB)',         'unit': 'g/L',   'type': 'continuous', 'min': 60.0,  'max': 200.0, 'default': 130.0, 'step': 0.1},
    'HbA1c':    {'label': 'HbA1c',                    'unit': '%',     'type': 'continuous', 'min': 4.0,   'max': 16.0,  'default': 7.0,   'step': 0.1},
    'HTN':      {'label': 'Hypertension (HTN)',        'unit': '',      'type': 'binary',     'options': {0: 'No', 1: 'Yes'}},
    'UA':       {'label': 'Uric Acid (UA)',            'unit': 'μmol/L','type': 'continuous', 'min': 100.0, 'max': 900.0, 'default': 320.0, 'step': 1.0},
    'sex':      {'label': 'Sex',                       'unit': '',      'type': 'binary',     'options': {0: 'Female (0)', 1: 'Male (1)'}},
    'MicroVCs': {'label': 'Microvascular Complications','unit': '',     'type': 'binary',     'options': {0: 'No', 1: 'Yes'}},
    'CVD':      {'label': 'Cardiovascular Disease (CVD)','unit': '',   'type': 'binary',     'options': {0: 'No', 1: 'Yes'}},
    'A/G':      {'label': 'Albumin/Globulin (A/G)',   'unit': '',      'type': 'continuous', 'min': 0.5,   'max': 3.0,   'default': 1.5,   'step': 0.01},
}


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem 0;">
        <div style="font-family: 'Space Mono', monospace; font-size: 0.6rem; 
                    letter-spacing: 0.2em; color: #00c9b1; text-transform: uppercase; 
                    margin-bottom: 0.25rem;">Clinical Tool</div>
        <div style="font-family: 'Crimson Pro', serif; font-size: 1.6rem; 
                    font-weight: 300; color: #f0eeea; line-height: 1.2;">
            Occult DKD<br><em style="color:#00c9b1;">Risk Predictor</em>
        </div>
    </div>
    <hr style="border-color: rgba(0,201,177,0.15); margin: 1rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Crimson Pro',serif; font-size:0.9rem; 
                color:#8899aa; line-height:1.7; padding-bottom:1rem;">
    This tool uses a validated Logistic Regression model to estimate the risk of 
    <strong style="color:#f0eeea;">occult diabetic kidney disease</strong> — 
    normal-range urinary albumin with subclinical renal dysfunction.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.6rem; 
                letter-spacing:0.15em; color:#00c9b1; text-transform:uppercase; 
                margin-bottom:0.5rem;">Model Specs</div>
    """, unsafe_allow_html=True)

    specs = {
        "Algorithm": "Logistic Regression",
        "Regularization": "L2 (C=1)",
        "Solver": "liblinear",
        "Features": "8 clinical variables",
    }
    for k, v in specs.items():
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; 
                    padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="font-family:'Crimson Pro',serif; font-size:0.85rem; color:#8899aa;">{k}</span>
            <span style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#f0eeea;">{v}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.55rem; 
                color:#556677; letter-spacing:0.05em; line-height:1.8;">
    ⚠️ FOR RESEARCH USE ONLY<br>
    Not a substitute for clinical judgment.<br>
    Always consult a qualified physician.
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">Occult <em>Diabetic Kidney Disease</em></h1>
    <p class="hero-subtitle">Clinlabomics · Machine Learning · Early Detection · Primary Care Screening</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🩺  RISK ASSESSMENT", "📊  MODEL INSIGHTS", "📖  ABOUT"])

# ─── TAB 1: Risk Assessment ───────────────────────────────────────────────────
with tab1:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown('<div class="section-label">Patient Parameters</div>', unsafe_allow_html=True)

        input_values = {}

        # Continuous Variables
        st.markdown("""
        <div style="font-family:'Space Mono',monospace; font-size:0.6rem; 
                    color:#8899aa; letter-spacing:0.1em; text-transform:uppercase; 
                    margin-bottom:0.75rem;">Laboratory Values</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            input_values['HGB'] = st.number_input(
                "Hemoglobin (g/L)",
                min_value=60.0, max_value=200.0, value=130.0, step=0.1,
                help="Hemoglobin concentration in g/L"
            )
        with c2:
            input_values['HbA1c'] = st.number_input(
                "HbA1c (%)",
                min_value=4.0, max_value=16.0, value=7.0, step=0.1,
                help="Glycated hemoglobin percentage"
            )

        c3, c4 = st.columns(2)
        with c3:
            input_values['UA'] = st.number_input(
                "Uric Acid (μmol/L)",
                min_value=100.0, max_value=900.0, value=320.0, step=1.0,
                help="Serum uric acid level"
            )
        with c4:
            input_values['A/G'] = st.number_input(
                "Albumin/Globulin Ratio",
                min_value=0.5, max_value=3.0, value=1.5, step=0.01,
                help="Albumin to globulin ratio"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Space Mono',monospace; font-size:0.6rem; 
                    color:#8899aa; letter-spacing:0.1em; text-transform:uppercase; 
                    margin-bottom:0.75rem;">Clinical Factors</div>
        """, unsafe_allow_html=True)

        c5, c6 = st.columns(2)
        with c5:
            sex_val = st.selectbox("Sex", options=[0, 1],
                                   format_func=lambda x: "Female" if x == 0 else "Male",
                                   help="0 = Female, 1 = Male")
            input_values['sex'] = sex_val

            htn_val = st.selectbox("Hypertension", options=[0, 1],
                                   format_func=lambda x: "No" if x == 0 else "Yes",
                                   help="History of hypertension")
            input_values['HTN'] = htn_val

        with c6:
            micro_val = st.selectbox("Microvascular Complications", options=[0, 1],
                                     format_func=lambda x: "No" if x == 0 else "Yes",
                                     help="Presence of microvascular complications")
            input_values['MicroVCs'] = micro_val

            cvd_val = st.selectbox("Cardiovascular Disease", options=[0, 1],
                                   format_func=lambda x: "No" if x == 0 else "Yes",
                                   help="History of cardiovascular disease")
            input_values['CVD'] = cvd_val

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⟶  CALCULATE RISK", type="primary"):
            st.session_state['calculated'] = True
            st.session_state['inputs'] = input_values.copy()

    # ─── Results Column ──────────────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="section-label">Risk Assessment</div>', unsafe_allow_html=True)

        if st.session_state.get('calculated', False):
            inputs = st.session_state['inputs']
            x = np.array([[inputs[f] for f in FEATURES]])
            
            import pandas as pd
            x_df = pd.DataFrame(x, columns=FEATURES)
            prob = model.predict_proba(x_df)[0][1]
            pred = model.predict(x_df)[0]

            # Risk level
            if prob >= 0.7:
                risk_label = "HIGH RISK"
                risk_color = "#e85d75"
                risk_bg = "result-high"
                interp = "This patient has an elevated probability of occult DKD. Consider further renal evaluation and close monitoring."
            elif prob >= 0.4:
                risk_label = "MODERATE RISK"
                risk_color = "#f4a535"
                risk_bg = "result-high"
                interp = "Intermediate risk detected. Clinical review and follow-up testing are advisable."
            else:
                risk_label = "LOW RISK"
                risk_color = "#00c9b1"
                risk_bg = "result-low"
                interp = "Low probability of occult DKD based on current parameters. Routine monitoring recommended."

            st.markdown(f"""
            <div class="{risk_bg}" style="margin-bottom:1.5rem;">
                <div class="result-label" style="color:{risk_color};">{risk_label}</div>
                <div class="result-value" style="color:{risk_color};">{prob:.1%}</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.6rem; 
                            color:rgba(255,255,255,0.4); margin-top:0.25rem; 
                            letter-spacing:0.1em;">PREDICTED PROBABILITY</div>
                <div class="result-interpretation">{interp}</div>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            fig, ax = plt.subplots(figsize=(5, 1.8), facecolor='none')
            ax.set_facecolor('none')
            fig.patch.set_alpha(0)

            # Background bar
            bar_bg = patches.FancyBboxPatch((0, 0.35), 1.0, 0.3,
                                             boxstyle="round,pad=0.01",
                                             facecolor='#1b2f44', edgecolor='none')
            ax.add_patch(bar_bg)

            # Color gradient zones
            zones = [(0, 0.4, '#00c9b1'), (0.4, 0.7, '#f4a535'), (0.7, 1.0, '#e85d75')]
            for start, end, color in zones:
                w = end - start
                z = patches.FancyBboxPatch((start, 0.36), w, 0.28,
                                            boxstyle="square,pad=0",
                                            facecolor=color, edgecolor='none', alpha=0.35)
                ax.add_patch(z)

            # Filled portion
            filled = patches.FancyBboxPatch((0, 0.36), prob, 0.28,
                                              boxstyle="square,pad=0",
                                              facecolor=risk_color, edgecolor='none', alpha=0.9)
            ax.add_patch(filled)

            # Needle/marker
            ax.plot([prob, prob], [0.25, 0.72], color='white', linewidth=2.5, solid_capstyle='round')
            ax.plot(prob, 0.72, 'o', color='white', markersize=7, zorder=5)

            # Labels
            for val, lbl in [(0, '0%'), (0.4, '40%'), (0.7, '70%'), (1.0, '100%')]:
                ax.text(val, 0.12, lbl, ha='center', va='top',
                        color='#8899aa', fontsize=7, fontfamily='monospace')

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Individual feature contributions
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Feature Contributions</div>', unsafe_allow_html=True)

            contributions = {f: COEF[i] * x[0][i] for i, f in enumerate(FEATURES)}
            sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

            fig2, ax2 = plt.subplots(figsize=(5, 3.5), facecolor='none')
            ax2.set_facecolor('none')
            fig2.patch.set_alpha(0)

            feature_labels = [f for f, _ in sorted_contrib]
            values = [v for _, v in sorted_contrib]
            colors = ['#e85d75' if v > 0 else '#00c9b1' for v in values]

            bars = ax2.barh(feature_labels, values, color=colors, alpha=0.85,
                            height=0.55, edgecolor='none')

            for bar, val in zip(bars, values):
                x_pos = val + (0.015 if val >= 0 else -0.015)
                ha = 'left' if val >= 0 else 'right'
                ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                         f'{val:+.3f}', ha=ha, va='center',
                         color='white', fontsize=7, fontfamily='monospace')

            ax2.axvline(0, color=(1, 1, 1, 0.2), linewidth=0.8, linestyle='--')
            ax2.set_facecolor('none')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_color('#2a4060')
            ax2.spines['bottom'].set_color('#2a4060')
            ax2.tick_params(colors='#8899aa', labelsize=8)
            ax2.set_xlabel('Contribution to log-odds', color='#8899aa', fontsize=8)
            ax2.yaxis.set_tick_params(labelcolor='#f0eeea')
            plt.tight_layout(pad=0.5)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        else:
            # Placeholder state
            st.markdown("""
            <div style="background: rgba(255,255,255,0.03); border: 1px dashed rgba(0,201,177,0.2);
                        border-radius: 16px; padding: 3rem 2rem; text-align: center; 
                        margin-top: 1rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem; opacity:0.4;">⟳</div>
                <div style="font-family:'Crimson Pro',serif; font-size:1.2rem; 
                            color:#8899aa; font-style:italic; line-height:1.7;">
                    Enter patient parameters and click<br>
                    <strong style="color:#f0eeea;">Calculate Risk</strong> to generate an assessment.
                </div>
                <div style="margin-top:1.5rem; font-family:'Space Mono',monospace; 
                            font-size:0.6rem; color:#445566; letter-spacing:0.1em;">
                    8-FEATURE LOGISTIC REGRESSION MODEL
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── TAB 2: Model Insights ────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label">Model Architecture</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace; font-size:0.6rem; 
                    color:#8899aa; letter-spacing:0.1em; text-transform:uppercase; 
                    margin-bottom:1rem;">Feature Coefficients</div>
        """, unsafe_allow_html=True)

        feature_display = {
            'HGB': 'Hemoglobin',
            'HbA1c': 'HbA1c',
            'HTN': 'Hypertension',
            'UA': 'Uric Acid',
            'sex': 'Sex',
            'MicroVCs': 'Microvascular Cx',
            'CVD': 'CVD',
            'A/G': 'Albumin/Globulin'
        }

        fig3, ax3 = plt.subplots(figsize=(5.5, 4), facecolor='none')
        ax3.set_facecolor('none')
        fig3.patch.set_alpha(0)

        sorted_idx = np.argsort(COEF)
        sorted_features = [feature_display[FEATURES[i]] for i in sorted_idx]
        sorted_coef = COEF[sorted_idx]
        bar_colors = ['#e85d75' if c > 0 else '#00c9b1' for c in sorted_coef]

        ax3.barh(sorted_features, sorted_coef, color=bar_colors, alpha=0.8,
                  height=0.6, edgecolor='none')
        ax3.axvline(0, color=(1, 1, 1, 0.15), linewidth=1)
        ax3.set_facecolor('none')
        for spine in ax3.spines.values():
            spine.set_color('#2a4060')
        ax3.tick_params(colors='#8899aa', labelsize=8)
        ax3.yaxis.set_tick_params(labelcolor='#f0eeea')
        ax3.set_xlabel('Coefficient (log-odds)', color='#8899aa', fontsize=8)
        ax3.set_title('Logistic Regression Coefficients', 
                       color='#f0eeea', fontsize=9, fontfamily='monospace', pad=12)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace; font-size:0.6rem; 
                    color:#8899aa; letter-spacing:0.1em; text-transform:uppercase; 
                    margin-bottom:1rem;">Coefficient Table</div>
        """, unsafe_allow_html=True)

        coef_data = pd.DataFrame({
            'Feature': [feature_display[f] for f in FEATURES],
            'Variable': FEATURES,
            'Coefficient': [f"{c:.4f}" for c in COEF],
            'Direction': ['↑ Risk' if c > 0 else '↓ Risk' for c in COEF],
        })

        st.dataframe(
            coef_data,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature", width="medium"),
                "Variable": st.column_config.TextColumn("Variable", width="small"),
                "Coefficient": st.column_config.TextColumn("Coef.", width="small"),
                "Direction": st.column_config.TextColumn("Effect", width="small"),
            }
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
            <p>
                <strong>Model intercept:</strong> {INTERCEPT:.4f}<br>
                The log-odds of occult DKD is computed as:
                <br><code style="color:#00c9b1; font-size:0.8rem;">
                log(p/1-p) = {INTERCEPT:.3f} + Σ(βᵢ·xᵢ)
                </code>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Probability Sensitivity</div>', unsafe_allow_html=True)

    # Sensitivity plot for continuous variables
    cont_features = ['HGB', 'HbA1c', 'UA', 'A/G']
    cont_defaults = [130.0, 7.0, 320.0, 1.5]
    cont_ranges = [(60, 200), (4, 16), (100, 900), (0.5, 3.0)]

    fig4, axes = plt.subplots(1, 4, figsize=(12, 3), facecolor='none')
    fig4.patch.set_alpha(0)

    default_input = {'HGB': 130.0, 'HbA1c': 7.0, 'HTN': 0, 'UA': 320.0,
                     'sex': 1, 'MicroVCs': 0, 'CVD': 0, 'A/G': 1.5}

    for idx, (feat, (lo, hi)) in enumerate(zip(cont_features, cont_ranges)):
        ax = axes[idx]
        ax.set_facecolor('none')

        x_range = np.linspace(lo, hi, 100)
        probs = []
        for val in x_range:
            inp = default_input.copy()
            inp[feat] = val
            x_arr = pd.DataFrame([[inp[f] for f in FEATURES]], columns=FEATURES)
            p = model.predict_proba(x_arr)[0][1]
            probs.append(p)

        # Fill under curve
        ax.fill_between(x_range, probs, alpha=0.15, color='#00c9b1')
        ax.plot(x_range, probs, color='#00c9b1', linewidth=2)
        ax.axhline(0.5, color='#f4a535', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1)
        ax.set_title(feat, color='#f0eeea', fontsize=8, fontfamily='monospace')
        ax.set_ylabel('P(DKD)' if idx == 0 else '', color='#8899aa', fontsize=7)
        ax.tick_params(colors='#8899aa', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('#2a4060')

    plt.suptitle('Risk Probability vs. Feature Value  (all other variables at default)',
                  color='#8899aa', fontsize=8, fontfamily='monospace', y=1.02)
    plt.tight_layout(pad=0.8)
    st.pyplot(fig4, use_container_width=True)
    plt.close()


# ─── TAB 3: About ─────────────────────────────────────────────────────────────
with tab3:
    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        st.markdown('<div class="section-label">Study Background</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Crimson Pro',serif; font-size:1.05rem; 
                    color:#c8d0d8; line-height:1.85;">
        <p>
        Diabetic kidney disease (DKD) is a leading cause of end-stage renal disease globally. 
        Conventional screening relies on urinary albumin-to-creatinine ratio (UACR), yet a 
        significant proportion of patients develop renal impairment with <em>normoalbuminuria</em> — 
        the so-called <strong style="color:#f0eeea;">occult DKD</strong> phenotype.
        </p>
        <p>
        This tool implements a clinlabomics-based machine learning model developed to enable 
        <strong style="color:#f0eeea;">early detection</strong> of occult DKD in primary care 
        settings, using only routinely-collected laboratory and clinical parameters.
        </p>
        <p>
        The model was trained on a cohort of type 2 diabetes patients and validated using 
        an external dataset, demonstrating robust discriminative performance.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">Feature Definitions</div>', unsafe_allow_html=True)
        defs = [
            ("HGB", "Hemoglobin", "g/L", "Anemia linked to early renal decline"),
            ("HbA1c", "Glycated Hemoglobin", "%", "Glycemic control marker"),
            ("HTN", "Hypertension", "Binary", "Key DKD risk accelerant"),
            ("UA", "Uric Acid", "μmol/L", "Associated with renal dysfunction"),
            ("sex", "Biological Sex", "0/1", "F=0, M=1"),
            ("MicroVCs", "Microvascular Cx", "Binary", "Retinopathy/neuropathy"),
            ("CVD", "Cardiovascular Dx", "Binary", "Comorbidity indicator"),
            ("A/G", "Albumin/Globulin", "Ratio", "Nutritional/hepatic status"),
        ]

        for code, name, unit, desc in defs:
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:0.75rem; 
                        padding:0.6rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="font-family:'Space Mono',monospace; font-size:0.65rem; 
                              color:#00c9b1; background:rgba(0,201,177,0.1); 
                              border:1px solid rgba(0,201,177,0.2); border-radius:4px;
                              padding:0.2rem 0.5rem; min-width:70px; text-align:center;
                              flex-shrink:0; margin-top:0.1rem;">{code}</span>
                <div>
                    <div style="font-family:'Crimson Pro',serif; font-size:0.95rem; 
                                color:#f0eeea; font-weight:600;">{name}
                        <span style="font-size:0.75rem; color:#8899aa; font-weight:400;"> · {unit}</span>
                    </div>
                    <div style="font-family:'Crimson Pro',serif; font-size:0.82rem; 
                                color:#8899aa; font-style:italic;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(232,93,117,0.08); border:1px solid rgba(232,93,117,0.25); 
                border-radius:12px; padding:1.25rem 1.5rem;">
        <div style="font-family:'Space Mono',monospace; font-size:0.65rem; 
                    color:#e85d75; letter-spacing:0.15em; text-transform:uppercase; 
                    margin-bottom:0.5rem;">⚠ Clinical Disclaimer</div>
        <div style="font-family:'Crimson Pro',serif; font-size:0.95rem; 
                    color:#c8d0d8; line-height:1.7;">
        This tool is intended for <strong style="color:#f0eeea;">research and educational purposes only</strong>. 
        It does not constitute medical advice and should not be used as the sole basis for clinical decisions. 
        All patient assessments must be performed by qualified healthcare professionals incorporating 
        the full clinical context.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; margin-top:2rem;
            border-top:1px solid rgba(255,255,255,0.06);">
    <span style="font-family:'Space Mono',monospace; font-size:0.6rem; 
                  color:#445566; letter-spacing:0.15em;">
        OCCULT DKD PREDICTOR · 8-FEATURE LR MODEL · FOR RESEARCH USE ONLY
    </span>
</div>
""", unsafe_allow_html=True)

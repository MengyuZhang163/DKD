# 🩺 Occult DKD Risk Predictor

A Streamlit web application for early detection of **occult diabetic kidney disease** using a validated 8-feature Logistic Regression model.

> Based on: *Developing and validating a clinlabomics-based machine learning model for early detection of occult diabetic kidney disease: Implications for primary care screening*

---

## 📋 Features

- **8-feature clinical model** using routinely available lab and clinical variables
- Real-time risk probability estimation
- Feature contribution visualization (per-patient log-odds breakdown)
- Sensitivity analysis plots for continuous variables
- Clean, dark-themed clinical UI built with Streamlit

### Input Variables

| Code | Variable | Type |
|------|----------|------|
| HGB | Hemoglobin | Continuous (g/L) |
| HbA1c | Glycated Hemoglobin | Continuous (%) |
| HTN | Hypertension | Binary (0/1) |
| UA | Uric Acid | Continuous (μmol/L) |
| sex | Biological Sex | Binary (0=F, 1=M) |
| MicroVCs | Microvascular Complications | Binary (0/1) |
| CVD | Cardiovascular Disease | Binary (0/1) |
| A/G | Albumin/Globulin Ratio | Continuous |

---

## 🚀 Deployment

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Streamlit Cloud (Recommended)

1. Fork or upload this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Deploy!

### Hugging Face Spaces

1. Create a new Space (type: Streamlit)
2. Upload all files from this repository
3. The app will auto-deploy

---

## 📁 File Structure

```
dkd_predictor/
├── app.py                 # Main Streamlit application
├── 8特征模型-LR.pkl        # Trained LR model (joblib format)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It does not constitute medical advice and should not replace clinical judgment. All patient assessments must be performed by qualified healthcare professionals.

---

## 🧬 Model Details

- **Algorithm**: Logistic Regression (scikit-learn)
- **Regularization**: L2, C=1
- **Solver**: liblinear
- **Random state**: 42
- **Features**: 8 clinlabomics variables
- **Output**: Probability of occult DKD (normoalbuminuric renal impairment)

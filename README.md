# 🧬 Breast Cancer Classification — Healthcare ML

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Healthcare](https://img.shields.io/badge/Domain-Healthcare-red?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

> A machine learning model to classify breast tumours as **Malignant** or **Benign** with high precision.

---

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide. Early and accurate detection is critical to improving patient outcomes. This project builds a classification model using clinical tumour measurements to predict whether a tumour is malignant (cancerous) or benign (non-cancerous).

> ⚕️ **Note:** In a medical context, minimising **false negatives** (missing a malignant tumour) is the top priority. This model is optimised for **high recall on the malignant class**.

---

## 🎯 Problem Statement

Given 30 clinical features (radius, texture, perimeter, area, smoothness, etc.) measured from a digitised image of a breast mass, predict whether the tumour is:
- 🔴 **Malignant (M)** — Cancerous
- 🟢 **Benign (B)** — Non-cancerous

---

## 📂 Dataset

- **Source:** Wisconsin Breast Cancer Dataset (UCI ML Repository / Kaggle)
- **Samples:** 569 patient records
- **Features:** 30 numerical features (mean, standard error, worst values)
- **Target:** Diagnosis — M (Malignant) or B (Benign)
- **Class Distribution:** 37.3% Malignant, 62.7% Benign

---

## 🔧 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML Models | Logistic Regression, SVM, Random Forest, Gradient Boosting |
| Data Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Model Evaluation | Scikit-learn metrics |

---

## 🚀 Methodology

### 1. Exploratory Data Analysis (EDA)
- Analysed class distribution and feature distributions
- Correlation heatmap to identify highly correlated features
- Box plots to visualise feature differences between M and B classes

### 2. Data Preprocessing
- Checked and handled missing values
- Label encoding for target variable
- Feature scaling using StandardScaler
- Train-test split (80:20) with stratification

### 3. Feature Selection
- Removed highly correlated features (correlation > 0.95)
- Selected top features using feature importance from Random Forest

### 4. Model Training & Tuning
- Trained Logistic Regression, SVM (RBF kernel), Random Forest, Gradient Boosting
- Hyperparameter tuning using GridSearchCV
- 5-fold cross-validation for robust evaluation

### 5. Evaluation Metrics
- Prioritised **Recall** (Sensitivity) for malignant class
- Confusion matrix, ROC-AUC curve, Precision-Recall curve

---

## 📊 Results

| Model | Accuracy | Precision | Recall (Malignant) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 97.4% | 96.8% | 97.6% | 97.2% | 0.99 |
| SVM (RBF) | 98.2% | 97.5% | 98.8% | 98.1% | 0.99 |
| Random Forest | 97.4% | 96.6% | 97.6% | 97.1% | 0.99 |
| Gradient Boosting | 97.4% | 97.6% | 96.5% | 97.0% | 0.99 |

✅ **Best Model: SVM (RBF Kernel)** — Selected for highest recall on malignant class.

---

## 📁 Project Structure

```
breast_cancer_classification/
│
├── breast_cancer_model.ipynb    # Main notebook
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies
```

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/Roumyajit/Data-Science-Projects.git
cd Data-Science-Projects

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Open the notebook
jupyter notebook breast_cancer_model.ipynb
```

---

## 🔮 Future Improvements

- [ ] Deploy as a Streamlit web app for clinical use
- [ ] Experiment with ensemble stacking for further accuracy gains
- [ ] Add SHAP explainability to explain individual predictions

---

## 👤 Author

**Roumyajit Sarkar**
- 📧 roumyajits@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/roumyajit-sarkar/)
- 🐙 [GitHub](https://github.com/Roumyajit)

---

⭐ If you found this project useful, please give it a star!

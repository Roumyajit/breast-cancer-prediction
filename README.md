# 🚀 Data Science Portfolio — Roumyajit Sarkar

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?style=flat&logo=scikit-learn)
![DL](https://img.shields.io/badge/Deep%20Learning-TensorFlow-red?style=flat&logo=tensorflow)
![NLP](https://img.shields.io/badge/NLP-NLTK-green?style=flat)
![Status](https://img.shields.io/badge/Projects-20%2B-brightgreen?style=flat)

> A collection of 20+ end-to-end Data Science and Machine Learning projects spanning NLP, Deep Learning, Healthcare, Sports Analytics, Business Intelligence, and more.

---

## 👤 About Me

Hi! I'm **Roumyajit Sarkar**, an aspiring Data Scientist currently pursuing M.Sc. in Data Science at Chandigarh University. I'm passionate about solving real-world problems using data, machine learning, and deep learning.

- 📧 **Email:** roumyajits@gmail.com
- 💼 **LinkedIn:** [linkedin.com/in/roumyajit-sarkar](https://www.linkedin.com/in/roumyajit-sarkar/)
- 🐙 **GitHub:** [github.com/Roumyajit](https://github.com/Roumyajit)

---

## 🗂️ Project Index

| # | Project | Domain | Tech Stack |
|---|---------|--------|------------|
| 01 | [Twitter Sentiment Analysis](#-1-twitter-sentiment-analysis) | NLP | NLTK, Scikit-learn |
| 02 | [Breast Cancer Classification](#-2-breast-cancer-classification) | Healthcare ML | Scikit-learn, SVM |
| 03 | [IPL Win Probability Prediction](#-3-ipl-win-probability-prediction) | Sports Analytics | Scikit-learn, Pandas |
| 04 | [Flight Price Prediction](#-4-flight-price-prediction) | Regression | Scikit-learn, Pandas |
| 05 | [Stock Price Prediction (LSTM)](#-5-stock-price-prediction--deep-learning) | Deep Learning | TensorFlow, Keras |
| 06 | [House Price Prediction](#-6-house-price-prediction) | Regression | Scikit-learn |
| 07 | [Diabetes Prediction](#-7-diabetes-prediction) | Healthcare ML | Scikit-learn |
| 08 | [Titanic Survival Prediction](#-8-titanic-survival-prediction) | Classification | Scikit-learn |
| 09 | [COVID-19 Data Analysis](#-9-covid-19-data-analysis) | EDA | Pandas, Matplotlib |
| 10 | [Air Quality Index Analysis](#-10-air-quality-index-analysis) | Environmental | Pandas, Seaborn |
| 11 | [IPL Data Analysis](#-11-ipl-data-analysis) | Sports EDA | Pandas, Matplotlib |
| 12 | [Netflix Content Analysis](#-12-netflix-content-analysis) | EDA | Pandas, Seaborn |
| 13 | [Netflix Viewing Analytics](#-13-netflix-viewing-analytics) | Business Analytics | Pandas, Matplotlib |
| 14 | [Uber NYC Trip Analysis](#-14-uber-nyc-trip-analysis) | Geospatial Analytics | Pandas, Seaborn |
| 15 | [YouTube Trending Analysis](#-15-youtube-trending-analysis) | Social Media Analytics | Pandas, Matplotlib |
| 16 | [Superstore Sales Analysis](#-16-superstore-sales-analysis) | Business Intelligence | Pandas, Seaborn |
| 17 | [E-commerce Analysis](#-17-e-commerce-analysis) | Business Analytics | Pandas, Matplotlib |
| 18 | [FIFA World Cup Analysis](#-18-fifa-world-cup-analysis) | Sports Analytics | Pandas, Seaborn |
| 19 | [Solar Energy Prediction](#-19-solar-energy-prediction) | Regression | Scikit-learn |
| 20 | [Rainfall Prediction](#-20-rainfall-prediction) | Regression | Scikit-learn |

---

## 📊 Featured Projects

---

### 🐦 1. Twitter Sentiment Analysis
**Domain:** NLP & Text Classification

Classifies tweet sentiments as Positive, Negative, or Neutral using an end-to-end NLP pipeline.

- Text preprocessing: tokenisation, stopword removal, stemming, lemmatisation
- TF-IDF vectorisation with unigrams and bigrams
- Compared Logistic Regression, Naive Bayes, and Linear SVM
- Visualised word clouds, confusion matrix, and sentiment distributions

**Tech Stack:** Python, NLTK, Scikit-learn, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./twitter_sentiment_analysis.ipynb)

---

### 🧬 2. Breast Cancer Classification
**Domain:** Healthcare Machine Learning

Predicts whether a breast tumour is Malignant or Benign using clinical measurements.

- Feature scaling, correlation-based feature selection
- Benchmarked Logistic Regression, SVM, Random Forest, Gradient Boosting
- Optimised for high Recall on malignant class (minimising false negatives)
- Evaluated using ROC-AUC, precision-recall curve, confusion matrix

**Tech Stack:** Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

🌐 **[Live Demo](https://roumyajit-breast-cancer-prediction.streamlit.app)** | 📄 [View Notebook](./breast_cancer_model.ipynb)

---

### 🏏 3. IPL Win Probability Prediction
**Domain:** Sports Analytics

Predicts live IPL match win probability ball-by-ball using historical match data.

- Engineered features: runs left, balls left, wickets in hand, CRR, RRR
- Trained Logistic Regression and Random Forest on 950+ IPL matches
- Produced win probability swing charts over full innings

**Tech Stack:** Python, Scikit-learn, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./ipl_advanced_analytics_win_probability.ipynb)

---

### ✈️ 4. Flight Price Prediction
**Domain:** Regression & Feature Engineering

Predicts flight ticket prices based on airline, route, stops, and booking details.

- Feature engineering: departure time bins, duration in minutes, airline encoding
- Compared Random Forest and Gradient Boosting regressors
- Reduced MAE through hyperparameter tuning

**Tech Stack:** Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

📄 [View Notebook](./flight_price_prediction.ipynb)

---

### 📈 5. Stock Price Prediction — Deep Learning
**Domain:** Time Series Forecasting

Forecasts stock prices using a stacked LSTM neural network.

- MinMax normalisation and sliding-window sequence construction
- Stacked LSTM with dropout regularisation and early stopping
- Evaluated using RMSE; visualised predicted vs actual price trends

**Tech Stack:** Python, TensorFlow, Keras, Pandas, NumPy, Matplotlib

📄 [View Notebook](./stock_price_prediction_dl.ipynb)

---

## 📁 All Projects

---

### 🏠 6. House Price Prediction
Regression model to predict real estate prices using property features.
Feature selection, model comparison, and residual analysis.

**Tech Stack:** Python, Scikit-learn, Pandas, Matplotlib

📄 [View Notebook](./house_price_prediction.ipynb)

---

### 💉 7. Diabetes Prediction
ML classification model to predict diabetes onset using the Pima Indians dataset.
Feature engineering, cross-validation, and model evaluation.

**Tech Stack:** Python, Scikit-learn, Pandas, Seaborn

📄 [View Notebook](./diabetes_prediction.ipynb)

---

### 🚢 8. Titanic Survival Prediction
End-to-end classification model with EDA, feature engineering (family size, title extraction), and multiple ML algorithms.

**Tech Stack:** Python, Scikit-learn, Pandas, Seaborn

📄 [View Notebook](./titanic_model.ipynb)

---

### 🦠 9. COVID-19 Data Analysis
Exploratory analysis of global COVID-19 trends with time-series visualisation of cases, deaths, and recovery rates.

**Tech Stack:** Python, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./covid19_analysis.ipynb)

---

### 🌍 10. Air Quality Index Analysis
Environmental data analysis of AQI trends across Indian cities. Pollution source identification using EDA and visualisations.

**Tech Stack:** Python, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./aqi_analysis.ipynb)

---

### 🏏 11. IPL Data Analysis
Exploratory analysis of IPL seasons — team performance, top batsmen, bowlers, and match outcome trends.

**Tech Stack:** Python, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./ipl_analysis.ipynb)

---

### 🎬 12. Netflix Content Analysis
EDA on Netflix catalogue — genre popularity, content type trends, and release year distributions.

**Tech Stack:** Python, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./netflix_analysis.ipynb)

---

### 📺 13. Netflix Viewing Analytics
Deep-dive into Netflix viewing patterns — watch time trends, popular genres, and binge-watching behaviour analysis.

**Tech Stack:** Python, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./netflix_viewing_analytics.ipynb)

---

### 🚖 14. Uber NYC Trip Analysis
Geospatial and temporal analysis of 4M+ Uber trips to identify demand hotspots, peak hours, and borough-level patterns.

**Tech Stack:** Python, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./uber_trip_analysis.ipynb)

---

### 📹 15. YouTube Trending Analysis
Analysis of trending video metadata to identify content engagement drivers, category trends, and regional patterns.

**Tech Stack:** Python, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./youtube_trending_analysis.ipynb)

---

### 🛍️ 16. Superstore Sales Analysis
Sales trend analysis, profitability insights, and customer segmentation from a retail superstore dataset.

**Tech Stack:** Python, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./superstore_sales_analysis.ipynb)

---

### 🛒 17. E-commerce Analysis
Customer purchase behaviour analysis and business insights from e-commerce transactional data.

**Tech Stack:** Python, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./ecommerce_analysis.ipynb)

---

### ⚽ 18. FIFA World Cup Analysis
Historical match data analysis revealing team performance patterns and tournament outcome trends across all World Cup editions.

**Tech Stack:** Python, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./fifa_world_cup_analysis.ipynb)

---

### ☀️ 19. Solar Energy Prediction
Regression model to forecast solar power output from weather and irradiance data.

**Tech Stack:** Python, Scikit-learn, Pandas, Matplotlib

📄 [View Notebook](./solar_energy_prediction.ipynb)

---

### 🌧️ 20. Rainfall Prediction
ML model to predict annual rainfall using monthly inputs from Indian meteorological data.

**Tech Stack:** Python, Scikit-learn, Pandas, Matplotlib, Seaborn

📄 [View Notebook](./rainfall_prediction.ipynb)

---

## 🛠️ Tech Stack Overview

| Category | Tools |
|---|---|
| **Languages** | Python, SQL |
| **ML & DL** | Scikit-learn, TensorFlow, Keras, XGBoost |
| **NLP** | NLTK, TF-IDF, Sentiment Analysis |
| **Data Analysis** | Pandas, NumPy, EDA, Feature Engineering |
| **Visualisation** | Matplotlib, Seaborn, Power BI, Tableau |
| **Big Data** | Apache Hadoop, Apache Spark |
| **Databases** | MySQL |
| **Tools** | Jupyter Notebook, Git, GitHub, Streamlit |

---

## 📜 Certifications

- 🏆 **Data Science & Machine Learning** — IIT Roorkee
- 📊 **Data Science** — Intellipaat
- 🤖 **Artificial Intelligence** — Intellipaat
- 🐍 **Python Programming** — Intellipaat
- 🗄️ **SQL & Microsoft SQL** — Intellipaat
- 📈 **Advanced Microsoft Excel** — Intellipaat

---

## 📫 Connect With Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/roumyajit-sarkar/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Roumyajit)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat&logo=gmail)](mailto:roumyajits@gmail.com)

---

⭐ **If you found my work useful, please star this repository!**

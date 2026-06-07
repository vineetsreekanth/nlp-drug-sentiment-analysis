# Drug Review Sentiment Analysis

NLP project on 215,063 real patient drug reviews — classifying sentiment as Positive, Neutral, or Negative using TF-IDF vectorisation and Machine Learning.

---

## Objective

Patients rate drugs on a scale of 1–10 and write free-text reviews. The goal is to train a model that predicts sentiment from the review text alone — without relying on the numeric rating.

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | UCI ML Repository / Kaggle |
| Total Reviews | 215,063 |
| Train Split | 161,297 |
| Test Split | 53,766 |
| Features | Drug name, condition, review text, rating (1–10), useful votes |

**Dataset link:** [UCI ML Drug Review Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018)

> Dataset not included in this repo due to file size. Download from the link above and place `drugsComTrain_raw.csv` and `drugsComTest_raw.csv` in the project folder.

---

## Workflow

1. Data Loading & Exploratory Data Analysis
2. Text Cleaning & Preprocessing
3. Sentiment Labeling (rating → Positive / Neutral / Negative)
4. TF-IDF Vectorization
5. Model Building — Logistic Regression & Naive Bayes
6. Evaluation, Insights & Conclusions

---

## Key Findings

- **Logistic Regression: 80.84% accuracy** — outperforms Naive Bayes (76.35%)
- Dataset is heavily skewed — 66% Positive, 25% Negative, 9% Neutral
- Neutral class underperforms (F1: 0.15) due to class imbalance
- **Positive reviews get 2× more helpful votes** (33.83) than negative ones (15.97) — counter-intuitive finding
- Top negative words: *worse, disappointed, horrible, ruined, useless*
- Top positive words: *love, amazing, changed life, miracle, lifesaver*

---

## Text Preprocessing Pipeline

Each review goes through 6 steps:
1. HTML decode
2. Remove HTML tags
3. Lowercase
4. Remove punctuation & numbers
5. Remove extra whitespace
6. Remove stopwords (NLTK)

---

## Model Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **80.84%** |
| Naive Bayes | 76.35% |

TF-IDF parameters: `max_features=10000`, `ngram_range=(1,2)`

Logistic Regression handles the high-dimensional sparse TF-IDF matrix better — it learns complex weight combinations across 10,000 features, whereas Naive Bayes assumes word independence which is too simplistic for nuanced medical language.

---

## Tools & Libraries

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, NLTK, re, html

---

## Files

| File | Description |
|------|-------------|
| `drug_review_sentiment_analysis.ipynb` | Full notebook with code, charts, and analysis |

---

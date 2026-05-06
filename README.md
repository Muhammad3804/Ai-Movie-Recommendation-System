# 🎬 CineAI — Hybrid AI Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

A Hybrid AI Movie Recommendation System built for the Artificial Intelligence course at **FAST-NUCES**. It chains together multiple AI techniques — constraint filtering, graph search, heuristic scoring, clustering, and a neural network — into one pipeline with a clean Streamlit UI.

---

## 🤖 AI Techniques Used

| Technique | Purpose |
|---|---|
| **CSP (Constraint Satisfaction)** | Filters movies by genre, year, runtime, rating, language |
| **BFS / DFS** | Basic graph search over the movie space |
| **A\* Search** | Optimal search using a genre + rating heuristic |
| **Heuristic Scoring** | Ranks candidates by rating, genre match, popularity, recency |
| **K-Means Clustering** | Groups movies into 8 taste clusters |
| **ANN (MLPRegressor)** | Predicts movie ratings from feature vectors |
| **Explainability Module** | Generates plain-English reasoning for each recommendation |

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/Muhammad3804/Ai-Movie-Recommendation-System.git
cd Ai-Movie-Recommendation-System
```

**2. Install dependencies**
```bash
pip install streamlit scikit-learn pandas numpy
```

**3. Add the dataset**

Download `tmdb_5000_movies.csv` from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place it in the root folder.

**4. Run the app**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📊 Dataset

- **Source:** [TMDB 5000 Movies — Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Size:** 4,803 movies
- **Key fields:** title, genres, release year, runtime, rating, popularity, language, overview

> ⚠️ The CSV file is not included in this repo. Download it from Kaggle and place it in the root folder before running.

---

## 📈 Results

Tested with Action + Drama, Year ≥ 2000, Rating ≥ 6.5:

- Movies after CSP filter: ~964
- A\* steps taken: ~33
- ANN R² Score: **0.933**

---

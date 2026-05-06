"""
ML Module
---------
1. K-means: clusters movies into groups based on features
2. ANN:     predicts a "predicted rating" for candidate movies
            given user preference vector
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


# ─── K-Means ─────────────────────────────────────────────────────────────────

def build_kmeans(df, n_clusters=8):
    """
    Clusters the full dataset based on: vote_average, popularity, runtime, year.
    Returns (kmeans model, scaler, cluster labels).
    """
    features = df[["vote_average", "popularity", "runtime", "year"]].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return km, scaler, labels


def get_cluster_for_movie(km, scaler, movie_row):
    x = scaler.transform([[
        movie_row["vote_average"],
        movie_row["popularity"],
        movie_row["runtime"],
        movie_row["year"] if movie_row["year"] > 0 else 2000
    ]])
    return int(km.predict(x)[0])


def get_cluster_movies(df, labels, cluster_id, top_n=10):
    cluster_df = df[labels == cluster_id].copy()
    return cluster_df.sort_values("weighted_rating", ascending=False).head(top_n)


# ─── ANN ─────────────────────────────────────────────────────────────────────

def build_feature_vector(row, all_genres):
    """Encode a movie row as a numeric vector for ANN."""
    genre_vec = [1 if g in row["genres_list"] else 0 for g in all_genres]
    return genre_vec + [
        row["vote_average"] / 10,
        min(row["popularity"] / 200, 1),
        min(row["runtime"] / 240, 1),
        (row["year"] - 1900) / 130 if row["year"] > 1900 else 0,
    ]


def train_ann(df, all_genres):
    """
    Train a simple ANN: input = movie feature vector, output = weighted_rating.
    Uses the real data so predictions stay grounded.
    """
    X, y = [], []
    for _, row in df.iterrows():
        try:
            fv = build_feature_vector(row, all_genres)
            X.append(fv)
            y.append(row["weighted_rating"])
        except:
            pass

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    ann = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=True,
    )
    ann.fit(X_train, y_train)
    test_score = ann.score(X_test, y_test)
    return ann, test_score


def predict_ratings(ann, df, candidate_indices, all_genres):
    """Returns dict {index: predicted_rating} for candidates."""
    preds = {}
    for idx in candidate_indices:
        row = df.iloc[idx]
        fv  = np.array(build_feature_vector(row, all_genres)).reshape(1, -1)
        preds[idx] = float(ann.predict(fv)[0])
    return preds

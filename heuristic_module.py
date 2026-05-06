"""
Heuristic Module — scores and ranks candidate movies.

Score = weighted combination of:
  - Normalised weighted rating
  - Genre match ratio
  - Popularity score (normalised)
  - Recency bonus (newer movies score slightly higher)
"""

import numpy as np


def score_movies(df, candidate_indices, preferred_genres, year_preference=None):
    """
    Returns list of (index, score, breakdown_dict) sorted by score descending.
    """
    candidates = df.iloc[candidate_indices].copy()

    # --- Normalise each component to [0, 1] ---
    max_rating = candidates["weighted_rating"].max() or 1
    max_pop    = candidates["popularity"].max() or 1
    max_year   = candidates["year"].max() or 1
    min_year   = candidates["year"].min() or 0

    results = []
    for idx in candidate_indices:
        row = df.iloc[idx]

        # 1. Rating score
        rating_score = row["weighted_rating"] / max_rating

        # 2. Genre match score
        movie_genres = set(row["genres_list"])
        pref_genres  = set(preferred_genres) if preferred_genres else set()
        if pref_genres:
            genre_score = len(movie_genres & pref_genres) / len(pref_genres)
        else:
            genre_score = 0.5

        # 3. Popularity score
        pop_score = row["popularity"] / max_pop

        # 4. Recency score
        year = row["year"] if row["year"] > 0 else min_year
        if max_year > min_year:
            recency_score = (year - min_year) / (max_year - min_year)
        else:
            recency_score = 0.5

        # Year preference bonus
        year_bonus = 0
        if year_preference and row["year"] > 0:
            year_bonus = max(0, 1 - abs(row["year"] - year_preference) / 20)

        # Final weighted score
        score = (
            0.40 * rating_score +
            0.30 * genre_score  +
            0.15 * pop_score    +
            0.10 * recency_score +
            0.05 * year_bonus
        )

        results.append((idx, score, {
            "rating_score":  round(rating_score,  3),
            "genre_score":   round(genre_score,   3),
            "pop_score":     round(pop_score,     3),
            "recency_score": round(recency_score, 3),
            "final_score":   round(score,         3),
        }))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

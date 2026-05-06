"""
Explanation Module — generates human-readable reasoning for recommendations.
"""

def explain(row, score_breakdown, predicted_rating, cluster_id,
            preferred_genres, constraints):
    reasons = []

    # Genre match
    matched = [g for g in preferred_genres if g in row["genres_list"]]
    if matched:
        reasons.append(f"✅ Matches your preferred genre(s): **{', '.join(matched)}**")

    # Rating
    rating = row["vote_average"]
    if rating >= 8.0:
        reasons.append(f"⭐ Critically acclaimed with a rating of **{rating}/10**")
    elif rating >= 7.0:
        reasons.append(f"⭐ Highly rated at **{rating}/10**")
    elif rating >= 6.0:
        reasons.append(f"⭐ Decent rating of **{rating}/10**")

    # ANN predicted rating
    if predicted_rating:
        reasons.append(f"🤖 ANN predicted rating: **{predicted_rating:.2f}/10**")

    # Cluster
    reasons.append(f"🔵 Grouped in movie cluster **#{cluster_id}** (similar taste group)")

    # Popularity
    pop = row["popularity"]
    if pop > 100:
        reasons.append(f"🔥 Very popular (popularity score: **{pop:.0f}**)")
    elif pop > 30:
        reasons.append(f"📈 Moderately popular")

    # Year
    year = row["year"]
    if year >= 2010:
        reasons.append(f"📅 Recent release: **{int(year)}**")
    elif year < 1990:
        reasons.append(f"🎬 Classic film from **{int(year)}**")

    # Heuristic score
    reasons.append(
        f"📊 Heuristic score: **{score_breakdown['final_score']}** "
        f"(rating: {score_breakdown['rating_score']}, "
        f"genre: {score_breakdown['genre_score']}, "
        f"popularity: {score_breakdown['pop_score']})"
    )

    # Active constraints
    if constraints:
        reasons.append(f"🔒 Passed CSP constraints: {' | '.join(constraints)}")

    return reasons

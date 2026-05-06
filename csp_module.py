"""
CSP Module — filters movies based on user constraints.

Variables:   each movie
Domains:     all movies in dataset
Constraints: genre, year range, runtime range, min rating
"""

def apply_csp(df, genres=None, year_min=None, year_max=None,
              runtime_min=None, runtime_max=None, min_rating=None,
              language=None):
    """
    Returns filtered DataFrame + a list of active constraint descriptions.
    """
    filtered = df.copy()
    active_constraints = []

    # Genre constraint
    if genres:
        def has_genre(g_list):
            return any(g in g_list for g in genres)
        filtered = filtered[filtered["genres_list"].apply(has_genre)]
        active_constraints.append(f"Genre ∈ {{{', '.join(genres)}}}")

    # Year range constraint
    if year_min is not None:
        filtered = filtered[filtered["year"] >= year_min]
        active_constraints.append(f"Year ≥ {year_min}")
    if year_max is not None:
        filtered = filtered[filtered["year"] <= year_max]
        active_constraints.append(f"Year ≤ {year_max}")

    # Runtime constraint
    if runtime_min is not None:
        filtered = filtered[filtered["runtime"] >= runtime_min]
        active_constraints.append(f"Runtime ≥ {runtime_min} min")
    if runtime_max is not None:
        filtered = filtered[filtered["runtime"] <= runtime_max]
        active_constraints.append(f"Runtime ≤ {runtime_max} min")

    # Minimum rating constraint
    if min_rating is not None:
        filtered = filtered[filtered["vote_average"] >= min_rating]
        active_constraints.append(f"Rating ≥ {min_rating}")

    # Language constraint
    if language and language != "Any":
        filtered = filtered[filtered["original_language"] == language]
        active_constraints.append(f"Language = {language}")

    return filtered.reset_index(drop=True), active_constraints

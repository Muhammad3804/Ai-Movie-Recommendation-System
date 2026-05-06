import pandas as pd
import numpy as np
import ast
import json

def load_movies(path="tmdb_5000_movies.csv"):
    df = pd.read_csv(path)

    def parse_json_col(val):
        try:
            items = ast.literal_eval(val)
            return [i["name"] for i in items if "name" in i]
        except:
            return []

    df["genres_list"]   = df["genres"].apply(parse_json_col)
    df["keywords_list"] = df["keywords"].apply(parse_json_col)

    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["runtime"]      = pd.to_numeric(df["runtime"],      errors="coerce").fillna(0)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)
    df["vote_count"]   = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0)
    df["popularity"]   = pd.to_numeric(df["popularity"],   errors="coerce").fillna(0)

    # weighted rating (IMDB formula)
    C = df["vote_average"].mean()
    m = df["vote_count"].quantile(0.25)
    df["weighted_rating"] = (
        (df["vote_count"] / (df["vote_count"] + m)) * df["vote_average"] +
        (m / (df["vote_count"] + m)) * C
    )

    df = df.dropna(subset=["title", "year"])
    df = df[df["runtime"] > 0]
    df = df.reset_index(drop=True)
    return df

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "Thriller", "War", "Western"
]

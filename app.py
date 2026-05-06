import streamlit as st
import pandas as pd
import numpy as np
import time

from data_loader  import load_movies, ALL_GENRES
from csp_module   import apply_csp
from search_module import build_genre_graph, bfs_search, dfs_search, astar_search
from heuristic_module import score_movies
from ml_module    import build_kmeans, get_cluster_for_movie, train_ann, predict_ratings
from explain_module import explain

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI — Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0a0a0f; color: #e8e8e8; }

.hero {
    background: linear-gradient(135deg, #0a0a0f 0%, #1a0a2e 50%, #0d1b2a 100%);
    border-bottom: 1px solid #2a2a3e;
    padding: 2rem 0 1.5rem;
    text-align: center;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    letter-spacing: 0.1em;
    background: linear-gradient(90deg, #e040fb, #00e5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub { color: #888; font-size: 0.9rem; letter-spacing: 0.2em; text-transform: uppercase; }

.card {
    background: #12121f;
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: #e040fb55; }

.movie-title { font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem; color: #fff; }
.movie-meta  { color: #888; font-size: 0.8rem; margin-bottom: 0.5rem; }
.badge {
    display: inline-block;
    background: #1e1e35;
    border: 1px solid #3a3a5e;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #aaa;
    margin-right: 4px;
    margin-bottom: 4px;
}
.score-pill {
    display: inline-block;
    background: linear-gradient(90deg, #e040fb22, #00e5ff22);
    border: 1px solid #e040fb55;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.85rem;
    color: #e040fb;
    font-weight: 500;
}
.reason { font-size: 0.83rem; color: #ccc; margin-bottom: 0.3rem; }
.section-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    letter-spacing: 0.15em;
    color: #00e5ff;
    border-bottom: 1px solid #1a2a3a;
    padding-bottom: 0.3rem;
    margin-bottom: 1rem;
}
.stat-box {
    background: #12121f;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.stat-num  { font-family: 'Bebas Neue'; font-size: 2rem; color: #e040fb; }
.stat-label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.1em; }
.algo-chip {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-right: 6px;
}
.chip-astar  { background: #1b2e1b; border: 1px solid #4caf5066; color: #81c784; }
.chip-bfs    { background: #1b2535; border: 1px solid #2196f366; color: #64b5f6; }
.chip-dfs    { background: #2e1b1b; border: 1px solid #f4433666; color: #e57373; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">🎬 CineAI</p>
  <p class="hero-sub">Hybrid AI Movie Recommendation System &nbsp;·&nbsp; CSP · Search · K-Means · ANN</p>
</div>
""", unsafe_allow_html=True)

# ─── Load & Cache Data ────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_movies("tmdb_5000_movies.csv")

@st.cache_resource
def get_ml_models(df):
    km, scaler, labels = build_kmeans(df, n_clusters=8)
    ann, r2 = train_ann(df, ALL_GENRES)
    return km, scaler, labels, ann, r2

df = get_data()
km, scaler, labels, ann, ann_r2 = get_ml_models(df)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">🎛️ User Preferences</p>', unsafe_allow_html=True)

    selected_genres = st.multiselect(
        "Preferred Genres",
        ALL_GENRES,
        default=["Action", "Adventure"]
    )

    st.markdown("**Year Range**")
    c1, c2 = st.columns(2)
    year_min = c1.number_input("From", min_value=1900, max_value=2025, value=2000, step=1)
    year_max = c2.number_input("To",   min_value=1900, max_value=2025, value=2024, step=1)

    st.markdown("**Runtime (minutes)**")
    r1, r2 = st.columns(2)
    rt_min = r1.number_input("Min", min_value=0,   max_value=300, value=60,  step=5)
    rt_max = r2.number_input("Max", min_value=30,  max_value=400, value=180, step=5)

    min_rating = st.slider("Minimum Rating ⭐", 0.0, 10.0, 6.0, 0.5)
    top_n      = st.slider("Top N Results",     3,   20,    8,   1)

    language = st.selectbox("Language", ["Any", "en", "fr", "es", "de", "ja", "ko", "hi"])
    algo     = st.selectbox("Search Algorithm", ["A* (Recommended)", "BFS", "DFS"])

    run = st.button("🎬 Get Recommendations", use_container_width=True)

# ─── Main Area ────────────────────────────────────────────────────────────────
if not run:
    st.markdown("""
    <br>
    <div style='text-align:center; color:#444; margin-top:4rem;'>
        <p style='font-size:3rem;'>🍿</p>
        <p style='font-family:Bebas Neue; font-size:1.8rem; color:#666; letter-spacing:0.1em;'>
            Configure your preferences in the sidebar and hit Recommend
        </p>
        <p style='color:#333; font-size:0.85rem;'>
            Using CSP · A* Search · K-Means Clustering · ANN · Heuristic Scoring
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Pipeline ─────────────────────────────────────────────────────────────────
with st.spinner("Running AI pipeline..."):
    t0 = time.time()

    # 1. CSP Filter
    filtered_df, constraints = apply_csp(
        df,
        genres=selected_genres if selected_genres else None,
        year_min=year_min, year_max=year_max,
        runtime_min=rt_min, runtime_max=rt_max,
        min_rating=min_rating,
        language=language if language != "Any" else None,
    )

    if len(filtered_df) < 3:
        st.error("⚠️ Too few movies match your constraints. Try relaxing some filters.")
        st.stop()

    # 2. Build graph on filtered set (cap at 500 for speed)
    cap = min(len(filtered_df), 500)
    graph_df = filtered_df.head(cap).reset_index(drop=True)
    adj, graph_df = build_genre_graph(graph_df, max_nodes=cap)

    # 3. Search
    start = 0
    if algo == "BFS":
        raw_results, steps = bfs_search(graph_df, adj, start, goal_rating=min_rating, max_results=top_n * 3)
        algo_name = "BFS"
    elif algo == "DFS":
        raw_results, steps = dfs_search(graph_df, adj, start, goal_rating=min_rating, max_results=top_n * 3)
        algo_name = "DFS"
    else:
        raw_results, steps = astar_search(graph_df, adj, start, preferred_genres=selected_genres, max_results=top_n * 3)
        algo_name = "A*"

    if not raw_results:
        st.warning("Search returned no results. Try different settings.")
        st.stop()

    # 4. Heuristic Scoring
    scored = score_movies(graph_df, raw_results, selected_genres)

    # 5. ANN predictions (map back to filtered_df indices)
    ann_preds = predict_ratings(ann, graph_df, raw_results, ALL_GENRES)

    # 6. K-Means cluster for each result
    cluster_map = {}
    for idx in raw_results:
        row = graph_df.iloc[idx]
        cluster_map[idx] = get_cluster_for_movie(km, scaler, row)

    elapsed = time.time() - t0

# ─── Stats Bar ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
s1, s2, s3, s4, s5 = st.columns(5)
with s1:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{len(df)}</div><div class="stat-label">Total Movies</div></div>', unsafe_allow_html=True)
with s2:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{len(filtered_df)}</div><div class="stat-label">After CSP Filter</div></div>', unsafe_allow_html=True)
with s3:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{steps}</div><div class="stat-label">Search Steps</div></div>', unsafe_allow_html=True)
with s4:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{ann_r2:.2f}</div><div class="stat-label">ANN R² Score</div></div>', unsafe_allow_html=True)
with s5:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{elapsed:.1f}s</div><div class="stat-label">Total Time</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Pipeline Trace ───────────────────────────────────────────────────────────
with st.expander("🔍 AI Pipeline Trace", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**CSP Constraints Applied:**")
        for c in constraints:
            st.markdown(f"- `{c}`")
        if not constraints:
            st.markdown("_No active constraints_")
    with col_b:
        chip = {"A*": "chip-astar", "BFS": "chip-bfs", "DFS": "chip-dfs"}[algo_name]
        st.markdown(f"""
        **Search Algorithm:** <span class="algo-chip {chip}">{algo_name}</span><br>
        **Graph nodes:** {cap} &nbsp;|&nbsp; **Steps taken:** {steps}<br>
        **Candidates found:** {len(raw_results)}<br>
        **K-Means clusters:** 8 &nbsp;|&nbsp; **ANN hidden layers:** 64→32
        """, unsafe_allow_html=True)

st.markdown('<p class="section-label">🏆 Top Recommendations</p>', unsafe_allow_html=True)

# ─── Results ─────────────────────────────────────────────────────────────────
shown = 0
for rank, (idx, score, breakdown) in enumerate(scored):
    if shown >= top_n:
        break
    row = graph_df.iloc[idx]
    pred_rating = ann_preds.get(idx, None)
    cluster_id  = cluster_map.get(idx, "?")
    reasons     = explain(row, breakdown, pred_rating, cluster_id, selected_genres, constraints)

    genres_html = "".join(f'<span class="badge">{g}</span>' for g in row["genres_list"])
    score_pct   = int(score * 100)

    with st.container():
        st.markdown(f"""
        <div class="card">
          <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
              <span class="movie-title">#{rank+1} &nbsp; {row['title']}</span><br>
              <span class="movie-meta">
                  {int(row['year'])} &nbsp;·&nbsp; {int(row['runtime'])} min &nbsp;·&nbsp;
                  ⭐ {row['vote_average']}/10 ({int(row['vote_count'])} votes)
              </span><br>
              {genres_html}
            </div>
            <div style="text-align:right;">
              <span class="score-pill">Match: {score_pct}%</span><br>
              <span style="font-size:0.78rem; color:#555; margin-top:4px; display:block;">
                  Cluster #{cluster_id} &nbsp;·&nbsp; Pred: {pred_rating:.1f}/10
              </span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if row.get("overview"):
            with st.expander("Overview & Reasoning", expanded=False):
                st.markdown(f"_{row['overview']}_")
                st.markdown("**Why this movie?**")
                for r in reasons:
                    st.markdown(f"<div class='reason'>{r}</div>", unsafe_allow_html=True)

    shown += 1

if shown == 0:
    st.warning("No movies could be scored. Try relaxing your constraints.")

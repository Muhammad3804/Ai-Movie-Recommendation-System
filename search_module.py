"""
Search Module — BFS, DFS, A* over the movie space.

State:        a movie index
Neighbours:   movies that share at least one genre (graph edges)
Goal:         movies with weighted_rating above threshold
Heuristic(A*): how well a movie matches user preferred genres (more shared = lower cost)
"""

from collections import deque
import heapq

def build_genre_graph(df, max_nodes=300):
    """Build adjacency list: movies connected if they share a genre."""
    df = df.head(max_nodes).reset_index(drop=True)
    n = len(df)
    adj = {i: [] for i in range(n)}

    for i in range(n):
        g_i = set(df.at[i, "genres_list"])
        for j in range(i + 1, n):
            g_j = set(df.at[j, "genres_list"])
            if g_i & g_j:  # shared genre = edge
                adj[i].append(j)
                adj[j].append(i)
    return adj, df


def bfs_search(df, adj, start_idx, goal_rating=6.5, max_results=20):
    """BFS from start node, collect movies meeting goal_rating."""
    visited = set()
    queue = deque([start_idx])
    results = []
    steps = 0

    while queue and len(results) < max_results:
        node = queue.popleft()
        steps += 1
        if node in visited:
            continue
        visited.add(node)

        if df.at[node, "vote_average"] >= goal_rating:
            results.append(node)

        for nb in adj[node]:
            if nb not in visited:
                queue.append(nb)

    return results, steps


def dfs_search(df, adj, start_idx, goal_rating=6.5, max_results=20):
    """DFS from start node, collect movies meeting goal_rating."""
    visited = set()
    stack = [start_idx]
    results = []
    steps = 0

    while stack and len(results) < max_results:
        node = stack.pop()
        steps += 1
        if node in visited:
            continue
        visited.add(node)

        if df.at[node, "vote_average"] >= goal_rating:
            results.append(node)

        for nb in reversed(adj[node]):
            if nb not in visited:
                stack.append(nb)

    return results, steps


def heuristic(df, node_idx, preferred_genres):
    """
    A* heuristic: lower value = better candidate.
    Based on genre overlap and inverse of weighted rating.
    """
    movie_genres = set(df.at[node_idx, "genres_list"])
    pref_genres  = set(preferred_genres)
    overlap      = len(movie_genres & pref_genres)
    rating       = df.at[node_idx, "weighted_rating"]

    # We want high overlap + high rating → low heuristic cost
    h = 10 - overlap - rating
    return h


def astar_search(df, adj, start_idx, preferred_genres, max_results=20):
    """
    A* search: priority = g(n) + h(n)
    g(n) = steps taken, h(n) = genre/rating heuristic
    """
    visited = set()
    # (f, g, node)
    heap = [(heuristic(df, start_idx, preferred_genres), 0, start_idx)]
    results = []
    steps = 0

    while heap and len(results) < max_results:
        f, g, node = heapq.heappop(heap)
        steps += 1
        if node in visited:
            continue
        visited.add(node)

        results.append(node)

        for nb in adj[node]:
            if nb not in visited:
                h = heuristic(df, nb, preferred_genres)
                heapq.heappush(heap, (g + 1 + h, g + 1, nb))

    return results, steps

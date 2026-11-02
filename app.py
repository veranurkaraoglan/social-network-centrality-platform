import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
from flask import Flask, jsonify


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
CACHE_DIR = BASE_DIR / "cache"
CACHE_FILE = CACHE_DIR / "dynamic_network_lab.json"

AIRPORTS_PATH = DATASET_DIR / "airports.csv"
ROUTES_PATH = DATASET_DIR / "routes.csv"

DECAY_FACTOR = 0.5
BETWEENNESS_K = 300
MAX_NODES = 300
MAX_EDGES = 1500
FORCE_INCLUDE_IATA = {"CDG"}

METRICS = [
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "harmonic",
    "katz",
    "pagerank",
    "percolation",
    "freeman",
    "decay",
]

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    min_v = min(scores.values())
    max_v = max(scores.values())
    if math.isclose(min_v, max_v):
        return {k: 0.0 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


def decay_chunk(args):
    graph_data, nodes, delta = args
    graph = nx.node_link_graph(graph_data, directed=True)
    out = {}
    for src in nodes:
        s = 0.0
        lengths = nx.single_source_shortest_path_length(graph, src)
        for dst, dist in lengths.items():
            if src == dst:
                continue
            s += delta**dist
        out[src] = s
    return out


def compute_decay_centrality(graph: nx.DiGraph, delta: float) -> Dict[str, float]:
    nodes = list(graph.nodes())
    if not nodes:
        return {}
    workers = min(max(os.cpu_count() or 1, 1), 6)
    if len(nodes) < 800 or workers == 1:
        out = {}
        for src in nodes:
            s = 0.0
            for dst, dist in nx.single_source_shortest_path_length(graph, src).items():
                if src == dst:
                    continue
                s += delta**dist
            out[src] = s
        return out

    chunk_size = math.ceil(len(nodes) / workers)
    chunks = [nodes[i : i + chunk_size] for i in range(0, len(nodes), chunk_size)]
    graph_data = nx.node_link_data(graph)
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for part in executor.map(decay_chunk, [(graph_data, c, delta) for c in chunks]):
            out.update(part)
    return out


def build_graph() -> Tuple[Dict[str, Dict], List[Tuple[str, str, int]], nx.DiGraph]:
    airports_df = pd.read_csv(AIRPORTS_PATH, low_memory=False)
    routes_df = pd.read_csv(ROUTES_PATH, low_memory=False)

    airports_df["Airport ID"] = airports_df["Airport ID"].astype(str)
    routes_df["Source airport ID"] = routes_df["Source airport ID"].astype(str)
    routes_df["Destination airport ID"] = routes_df["Destination airport ID"].astype(str)

    airports_df = airports_df[airports_df["Airport ID"] != r"\N"].copy()
    routes_df = routes_df[
        (routes_df["Source airport ID"] != r"\N")
        & (routes_df["Destination airport ID"] != r"\N")
    ].copy()

    valid_airports = set(airports_df["Airport ID"].tolist())
    routes_df = routes_df[
        routes_df["Source airport ID"].isin(valid_airports)
        & routes_df["Destination airport ID"].isin(valid_airports)
    ].copy()

    airports = {}
    for _, row in airports_df.iterrows():
        airports[row["Airport ID"]] = {
            "id": row["Airport ID"],
            "name": row.get("Name", "Unknown Airport"),
            "city": row.get("City", ""),
            "country": row.get("Country", ""),
            "iata": row.get("IATA", ""),
            "icao": row.get("ICAO", ""),
            "latitude": float(row.get("Latitude", 0.0)),
            "longitude": float(row.get("Longitude", 0.0)),
        }

    weighted_routes = (
        routes_df.groupby(["Source airport ID", "Destination airport ID"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
    )
    edges = list(weighted_routes[["Source airport ID", "Destination airport ID"]].itertuples(index=False, name=None))
    graph = nx.DiGraph()
    graph.add_nodes_from(airports.keys())
    graph.add_edges_from(edges)

    # keep graph renderable on browser, using top-degree backbone
    degree_all = nx.degree_centrality(graph)
    selected_nodes = [
        node for node, _ in sorted(degree_all.items(), key=lambda x: x[1], reverse=True)[:MAX_NODES]
    ]
    selected_set = set(selected_nodes)
    force_ids = {
        airport_id
        for airport_id, info in airports.items()
        if str(info.get("iata", "")).upper() in FORCE_INCLUDE_IATA
    }
    selected_set.update(force_ids)
    selected_routes = weighted_routes[
        weighted_routes["Source airport ID"].isin(selected_set)
        & weighted_routes["Destination airport ID"].isin(selected_set)
    ].copy()

    selected_routes = selected_routes.sort_values("weight", ascending=False).head(MAX_EDGES)
    filtered_edges = list(
        selected_routes[["Source airport ID", "Destination airport ID", "weight"]].itertuples(
            index=False, name=None
        )
    )

    filtered_graph = nx.DiGraph()
    filtered_graph.add_nodes_from(selected_nodes)
    filtered_graph.add_weighted_edges_from(filtered_edges)

    # preserve network integrity by keeping the largest weakly connected component
    if filtered_graph.number_of_nodes() > 0 and filtered_graph.number_of_edges() > 0:
        giant_nodes = max(nx.weakly_connected_components(filtered_graph), key=len)
    else:
        giant_nodes = set(filtered_graph.nodes())

    filtered_graph = filtered_graph.subgraph(giant_nodes).copy()
    filtered_edges = [
        (u, v, int(data.get("weight", 1))) for u, v, data in filtered_graph.edges(data=True)
    ]
    filtered_airports = {k: v for k, v in airports.items() if k in giant_nodes}
    return filtered_airports, filtered_edges, filtered_graph


def compute_metrics(graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    n = graph.number_of_nodes()
    k = min(max(20, BETWEENNESS_K), max(1, n))
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, k=k, normalized=True, seed=42)
    closeness = nx.closeness_centrality(graph)
    harmonic = nx.harmonic_centrality(graph)
    pagerank = nx.pagerank(graph, alpha=0.85, tol=1e-06)

    try:
        eigenvector = nx.eigenvector_centrality(graph, max_iter=2000, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        eigenvector = nx.eigenvector_centrality_numpy(graph)

    try:
        katz = nx.katz_centrality(graph, alpha=0.005, beta=1.0, max_iter=2000, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        katz = nx.katz_centrality_numpy(graph, alpha=0.005, beta=1.0)

    decay = compute_decay_centrality(graph, DECAY_FACTOR)

    # Node-level proxy derived from Freeman centralization concept.
    max_degree = max(degree.values()) if degree else 0.0
    freeman = {node: max_degree - value for node, value in degree.items()}

    states = {node: degree.get(node, 0.0) for node in graph.nodes()}
    try:
        percolation = nx.percolation_centrality(graph.to_undirected(), states=states)
    except Exception:
        percolation = {node: 0.0 for node in graph.nodes()}

    return {
        "degree": degree,
        "betweenness": betweenness,
        "closeness": closeness,
        "eigenvector": eigenvector,
        "harmonic": harmonic,
        "katz": katz,
        "pagerank": pagerank,
        "percolation": percolation,
        "freeman": freeman,
        "decay": decay,
    }


def build_payload() -> Dict:
    airports, edges, graph = build_graph()
    raw_metrics = compute_metrics(graph)
    normalized_metrics = {name: normalize(values) for name, values in raw_metrics.items()}

    nodes = []
    for node_id, airport in airports.items():
        nodes.append(
            {
                **airport,
                "metrics": {m: float(raw_metrics[m].get(node_id, 0.0)) for m in METRICS},
                "metricsNormalized": {
                    m: float(normalized_metrics[m].get(node_id, 0.0)) for m in METRICS
                },
            }
        )
    edges_payload = [{"source": s, "target": t, "weight": w} for s, t, w in edges]
    return {
        "meta": {
            "nodeCount": graph.number_of_nodes(),
            "edgeCount": graph.number_of_edges(),
            "metrics": METRICS,
        },
        "nodes": nodes,
        "edges": edges_payload,
    }


def load_or_build(force: bool = False) -> Dict:
    if CACHE_FILE.exists() and not force:
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    with CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    return payload


@app.get("/api/network-lab/health")
def health():
    return jsonify({"ok": True})


@app.get("/api/network-lab/data")
def data():
    return jsonify(load_or_build(force=False))


@app.post("/api/network-lab/rebuild")
def rebuild():
    payload = load_or_build(force=True)
    return jsonify({"ok": True, "meta": payload["meta"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

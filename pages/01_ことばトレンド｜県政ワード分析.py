import os
import io
import re
import json
import urllib.request
from collections import Counter, defaultdict
import pandas as pd

import streamlit as st
import boto3
import zstandard as zstd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import itertools
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from networkx.algorithms.community import greedy_modularity_communities

import json as _json
import math


st.set_page_config(page_title="ことばトレンド県政", layout="wide", page_icon="⚖️")

# ========================
# 設定
# ========================
# S3の場所
S3_BUCKET = st.secrets["AWS-KEY"]["DATA_BUCKET_NAME"]
S3_KEY    = "pref_yamaguchi/trending-words/pref-yamaguchi.jsonl.zst"
AWS_REGION = "us-west-2"
AWS_ACCESS_KEY = st.secrets["AWS-KEY"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS-KEY"]["AWS_SECRET_KEY"]

def make_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )

# ========================
# フォント（日本語表示用）
# ========================
def get_font_path():
    font_dir = "./font"
    os.makedirs(font_dir, exist_ok=True)
    font_path = os.path.join(font_dir, "NotoSansCJKjp-Regular.otf")
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
        urllib.request.urlretrieve(url, font_path)
    return font_path

# ========================
# データ読み込み（S3のJSONL.zst）
# ========================
@st.cache_data(ttl=1800, show_spinner=False)  # 30分キャッシュ
def load_preagg_records(bucket: str, key: str):
    s3 = make_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]  # StreamingBody

    dctx = zstd.ZstdDecompressor()
    records = []
    with dctx.stream_reader(body) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="strict", newline="")
        for line in text_stream:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records

# ========================
# ワードクラウド
# ========================
def draw_wordcloud(freq: dict):
    if not freq:
        st.info("対象条件に一致する語句がありませんでした。")
        return
    font_path = get_font_path()
    wc = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color="white",
        colormap="tab10",
    )
    wc.generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# ========================
# 合算ユーティリティ
# ========================
def aggregate_terms(selected_records):
    counter = Counter()
    total_utter = 0
    for r in selected_records:
        total_utter += int(r.get("utterances", 0))
        for term, cnt in r.get("top_terms", []):
            counter[term] += int(cnt)
    return counter, total_utter

# ========================
# 共起ネットワーク構築
# ========================
@st.cache_data(ttl=1800, show_spinner=False)
def build_cooccurrence(records, global_freq: Counter, *,
                       top_k_per_doc: int = 30,
                       max_nodes_global: int = 80,
                       weight_mode: str = "binary"):  # "binary" or "mincnt"
    allowed_terms = set([t for t, _ in global_freq.most_common(max_nodes_global)])

    edge_weight = Counter()
    node_weight = Counter()

    for r in records:
        terms = r.get("top_terms", [])[:top_k_per_doc]
        terms = [(t, int(c)) for t, c in terms if t in allowed_terms]

        for t, c in terms:
            node_weight[t] += c

        for (t1, c1), (t2, c2) in itertools.combinations(terms, 2):
            a, b = sorted((t1, t2))
            if weight_mode == "mincnt":
                edge_weight[(a, b)] += min(c1, c2)
            else:
                edge_weight[(a, b)] += 1

    G = nx.Graph()
    for t, w in node_weight.items():
        G.add_node(t, size=w)
    for (a, b), w in edge_weight.items():
        G.add_edge(a, b, weight=w)
    return G

# ========================
# 共起ネットワークをpyvisで可視化
# ========================
def render_pyvis_network(
    G: nx.Graph,
    *,
    min_edge_weight: int = 2,
    physics: bool = False,
    height_px: int = 720,
    label_font_size: int = 22,
    enable_clustering: bool = True,
    focus_community: int | None = None,
    drop_isolates: bool = True,
    keep_largest_component: bool = False
):
    H = nx.Graph()
    for n, attrs in G.nodes(data=True):
        H.add_node(n, **attrs)
    for u, v, attrs in G.edges(data=True):
        if int(attrs.get("weight", 1)) >= min_edge_weight:
            H.add_edge(u, v, **attrs)

    if drop_isolates:
        H.remove_nodes_from(list(nx.isolates(H)))

    if keep_largest_component and H.number_of_nodes() > 0:
        comps = list(nx.connected_components(H))
        giant = max(comps, key=len)
        H = H.subgraph(giant).copy()

    comm_map, comms = ({}, [])
    if enable_clustering and H.number_of_nodes() > 0:
        comm_map, comms = detect_communities(H)

    if focus_community is not None and enable_clustering and comms:
        keep = comms[focus_community] if focus_community < len(comms) else set()
        H = H.subgraph(keep).copy()
        if H.number_of_nodes() > 0:
            comm_map, comms = detect_communities(H)

    pos = layout_communities_with_warmstart(
        H, comms,
        cluster_k=2.0,
        local_k=0.5,
        cluster_scale=900,
        local_scale=550,
        seed=42
    )

    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#1f2937",
        notebook=False,
        directed=False,
    )

    for n, attrs in H.nodes(data=True):
        val = int(attrs.get("size", 1))
        group = comm_map.get(n) if enable_clustering else None
        x, y = pos.get(n, (None, None))
        net.add_node(
            n,
            label=n,
            va

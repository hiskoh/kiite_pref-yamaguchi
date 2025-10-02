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
S3_BUCKET =  st.secrets["AWS-KEY"]["DATA_BUCKET_NAME"]
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
# ワードクラウド描画
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
                       weight_mode: str = "binary"  # "binary" or "mincnt"
                       ):
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
            value=val,
            title=f"{n}: {val}",
            group=group,
            x=x, y=y,
            physics=physics,
            fixed=not physics
        )

    for u, v, attrs in H.edges(data=True):
        w = int(attrs.get("weight", 1))
        net.add_edge(u, v, value=w, title=f"co-occur: {w}")

    options = {
        "nodes": {
            "font": {"size": label_font_size, "face": "sans-serif"},
            "scaling": {
                "min": 10, "max": 50,
                "label": {"enabled": True,
                          "min": max(10, label_font_size-6),
                          "max": label_font_size+10}
            }
        },
        "edges": {"smooth": False},
        "interaction": {"tooltipDelay": 100, "hover": True},
        "physics": {
            "enabled": physics,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "springLength": 120,
                "springConstant": 0.08,
                "avoidOverlap": 0.6,
                "damping": 0.45
            },
            "stabilization": {"enabled": physics, "iterations": 300},
            "minVelocity": 0.5,
            "maxVelocity": 30
        }
    }
    net.set_options(_json.dumps(options))
    html = net.generate_html(notebook=False)
    components.html(html, height=height_px, scrolling=True)
    return comms

def community_meta_graph(H: nx.Graph, comms):
    M = nx.Graph()
    for i, nodes in enumerate(comms):
        M.add_node(i, size=len(nodes))
    for u, v, d in H.edges(data=True):
        cu = next(i for i, S in enumerate(comms) if u in S)
        cv = next(i for i, S in enumerate(comms) if v in S)
        if cu == cv:
            continue
        w = int(d.get("weight", 1))
        if M.has_edge(cu, cv):
            M[cu][cv]["weight"] += w
        else:
            M.add_edge(cu, cv, weight=w)
    return M

def layout_communities_with_warmstart(
    H: nx.Graph,
    comms,
    *,
    cluster_k: float = 2.0,
    local_k: float = 0.5,
    cluster_scale: float = 900,
    local_scale: float = 550,
    seed: int = 42
):
    if not comms:
        return nx.spring_layout(H, k=local_k, weight="weight", seed=seed)

    M = community_meta_graph(H, comms)
    pos_comm = nx.spring_layout(M, k=cluster_k, weight="weight", seed=seed)

    pos = {}
    for i, nodes in enumerate(comms):
        sub = H.subgraph(nodes)
        local = nx.spring_layout(sub, k=local_k, weight="weight", seed=seed)
        cx, cy = pos_comm.get(i, (0.0, 0.0))
        cx *= cluster_scale
        cy *= cluster_scale
        for n, (x, y) in local.items():
            pos[n] = (cx + x * local_scale, cy + y * local_scale)
    return pos

def detect_communities(G: nx.Graph):
    if G.number_of_nodes() == 0:
        return {}, []
    comms = list(greedy_modularity_communities(G, weight="weight"))
    comm_map = {}
    for i, s in enumerate(comms):
        for n in s:
            comm_map[n] = i
    return comm_map, comms

def summarize_communities(G: nx.Graph, comms):
    summaries = []
    for i, nodes in enumerate(comms):
        rows = sorted(
            [(n, int(G.nodes[n].get("size", 1))) for n in nodes],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        summaries.append({"community": i, "top_terms": rows, "size": len(nodes)})
    return summaries

def layout_by_community(H: nx.Graph, comms, *, intra_k=0.5, spacing=800, seed=42):
    if not comms:
        return nx.spring_layout(H, k=intra_k, weight="weight", seed=seed)

    pos = {}
    n_comm = len(comms)
    for i, nodes in enumerate(comms):
        angle = 2 * math.pi * i / n_comm
        cx = spacing * math.cos(angle)
        cy = spacing * math.sin(angle)
        sub = H.subgraph(nodes)
        local = nx.spring_layout(sub, k=intra_k, weight="weight", seed=seed)
        for n, (x, y) in local.items():
            pos[n] = (x * 300 + cx, y * 300 + cy)
    return pos

def layout_by_community_grid(H: nx.Graph, comms, *,
                             cluster_spacing=1200,
                             subgraph_scale=500,
                             grid_cols=3,
                             seed=42):
    if not comms:
        return nx.spring_layout(H, k=0.5, weight="weight", seed=seed)
    pos = {}
    for i, nodes in enumerate(comms):
        sub = H.subgraph(nodes)
        local = nx.spring_layout(sub, k=0.5, weight="weight", seed=seed)
        row, col = divmod(i, grid_cols)
        cx = col * cluster_spacing
        cy = row * cluster_spacing
        for n, (x, y) in local.items():
            pos[n] = (x * subgraph_scale + cx, y * subgraph_scale + cy)
    return pos

def _edge_count_at_threshold(G: nx.Graph, thr: int) -> int:
    return sum(1 for _, _, d in G.edges(data=True) if int(d.get("weight", 1)) >= thr)

def recommend_min_edge(G: nx.Graph, start_thr: int, *, target_max_edges: int, cap: int,
                       step_back_if_below: bool = True) -> tuple[int, int]:
    thr = start_thr
    edges = _edge_count_at_threshold(G, thr)
    while edges > target_max_edges and thr < cap:
        thr += 1
        edges = _edge_count_at_threshold(G, thr)
    if thr > start_thr and edges < target_max_edges * 0.6:
        prev_edges = _edge_count_at_threshold(G, thr - 1)
        return thr - 1, prev_edges
    return thr, edges

def _mark_min_edge_touched():
    st.session_state["min_edge_user_touched"] = True

def parse_date_from_chunk_head(chunk_id: str) -> pd.Timestamp | None:
    if not chunk_id:
        return None
    head = str(chunk_id).split("_", 1)[0]
    m = re.match(r"^\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", head)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    try:
        return pd.Timestamp(y, mo, d)
    except Exception:
        return None

# ========================
# UI 本体
# ========================
st.title("⚖️ことばトレンド｜県政ワード分析")
st.markdown("""
<span style="color:#6b7280">
発言者や会議などを選んで、発言の傾向をワードクラウドやネットワークで可視化します。<br>
年度や会議、発言者を変えて、発言の傾向や特徴を見てみましょう。
</span>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stMultiSelect [data-baseweb="tag"]{
  background-color:#eef3f8 !important;
  color:#1f2937 !important;
  border:1px solid #d1d5db !important;
}
.stMultiSelect [data-baseweb="tag"] [data-baseweb="tag-close-icon"]{
  color:#6b7280 !important;
}
</style>
""", unsafe_allow_html=True)

with st.spinner("データを読み込んでいます…"):
    try:
        data = load_preagg_records(S3_BUCKET, S3_KEY)
    except Exception as e:
        st.error("サーバーからの読込に失敗しました。時間をおいて再度お試しください。")
        st.stop()

if not data:
    st.warning("事前集計データが空でした。")
    st.stop()

# --------- ここで session_state を初期化（NameError/KeyError回避）---------
DEFAULT_TOPK = 30
DEFAULT_MAXN = 80
DEFAULT_MINEDGE = 2
DEFAULT_LABEL_FONT = 22
DEFAULT_WEIGHT_MODE = "binary"
DEFAULT_PHYSICS = True

ss = st.session_state
ss.setdefault("top_k_per_doc", DEFAULT_TOPK)
ss.setdefault("max_nodes_global", DEFAULT_MAXN)
ss.setdefault("min_edge_weight", DEFAULT_MINEDGE)
ss.setdefault("weight_mode", DEFAULT_WEIGHT_MODE)
ss.setdefault("label_font_size", DEFAULT_LABEL_FONT)
ss.setdefault("physics_on", DEFAULT_PHYSICS)
ss.setdefault("min_edge_user_touched", False)
ss.setdefault("auto_min_edge", True)
ss.setdefault("target_max_edges", 200)
ss.setdefault("min_edge_cap", 20)

# ========= 役職の限定（知事・委員のみ） + 会議名プレフィックス/種別を付与 =========
ALLOWED_ROLES = {"知事", "委員"}

def get_meeting_prefix(name: str | None) -> str | None:
    if not name:
        return None
    # 全角ハイフンで分割、手前を採用
    return str(name).split("－")[0].strip()

def classify_meeting_type(name: str | None) -> str | None:
    """'委員会' を含むかどうかで分類。含めば '委員会'、それ以外は '定例会・臨時会'。"""
    if not name:
        return None
    s = str(name)
    return "委員会" if "委員会" in s else "定例会・臨時会"

for r in data:
    r["_meeting_prefix"] = get_meeting_prefix(r.get("meeting_name"))
    r["_meeting_type"]   = classify_meeting_type(r.get("meeting_name"))

# ========= フィルタ用ユニーク値（知事・委員のみを対象）=========
years = sorted({
    r.get("year")
    for r in data
    if r.get("speaker_role") in ALLOWED_ROLES and r.get("year")
})
speakers = sorted({
    r.get("speaker")
    for r in data
    if r.get("speaker_role") in ALLOWED_ROLES and r.get("speaker")
})
meeting_types = [t for t in ["定例会・臨時会", "委員会"]
                 if any(r.get("speaker_role") in ALLOWED_ROLES and r.get("_meeting_type")==t for r in data)]
meeting_prefixes_all = sorted({
    r.get("_meeting_prefix")
    for r in data
    if r.get("speaker_role") in ALLOWED_ROLES and r.get("_meeting_prefix")
})

# ========= 役職セレクタは非表示。1行で（発言者／年／会議種別／会議名） =========
st.divider()
st.subheader("条件")

def expand_all(selected, universe):
    if ("すべて" in selected) or (not selected):
        return set(universe)
    return set(selected)

c1, c2, c3, c4 = st.columns(4)

with c1:
    speaker_options = ["すべて"] + speakers
    current_speaker = st.session_state.get("speaker_sb", "すべて")
    if current_speaker not in speaker_options:
        current_speaker = "すべて"
    sel_speaker = st.selectbox("発言者", speaker_options,
                               index=speaker_options.index(current_speaker),
                               key="speaker_sb")

with c2:
    sel_years = st.multiselect("発言年", ["すべて"] + years, default=["すべて"])

with c3:
    sel_meeting_types = st.multiselect("会議種別", ["すべて"] + meeting_types, default=["定例会・臨時会"])

# 会議名（プレフィックス）は、選択された会議種別に応じて候補を絞る
if ("すべて" in sel_meeting_types) or (not sel_meeting_types):
    meeting_prefix_candidates = meeting_prefixes_all
else:
    selected_types = set(sel_meeting_types)
    meeting_prefix_candidates = sorted({
        r.get("_meeting_prefix")
        for r in data
        if (r.get("speaker_role") in ALLOWED_ROLES)
        and (r.get("_meeting_type") in selected_types)
        and r.get("_meeting_prefix")
    })

with c4:
    sel_meet_prefix = st.multiselect("会議名",
                                     ["すべて"] + meeting_prefix_candidates,
                                     default=["すべて"])

years_set = expand_all(sel_years, years)
meeting_types_set = expand_all(sel_meeting_types, meeting_types)
meet_prefix_set = expand_all(sel_meet_prefix, meeting_prefix_candidates)

# 発言者は単一選択（“すべて” は全員許容）
speaker_set = set(speakers) if sel_speaker == "すべて" else {sel_speaker}

# ========= 実フィルタ（役職限定 + 年 + 種別 + 会議名プレフィックス + 発言者）=========
filtered = [
    r for r in data
    if (r.get("speaker_role") in ALLOWED_ROLES)
    and (r.get("year") in years_set)
    and (r.get("_meeting_type") in meeting_types_set)
    and (r.get("_meeting_prefix") in meet_prefix_set)
    and (r.get("speaker") in speaker_set)
]

# ===== ▼ワードクラウド 画像表示部 =====
st.divider()
st.subheader("頻出単語")

freq_counter, total_utterances = aggregate_terms(filtered)
draw_wordcloud(freq_counter)

# ===== ▼ネットワーク 表示部 =====
G = build_cooccurrence(
    filtered,
    freq_counter,
    top_k_per_doc=ss["top_k_per_doc"],
    max_nodes_global=ss["max_nodes_global"],
    weight_mode=ss["weight_mode"],
)

if (
    ss.get("auto_min_edge", True) and
    not ss["min_edge_user_touched"] and
    ss["min_edge_weight"] == DEFAULT_MINEDGE
):
    new_thr, edge_cnt = recommend_min_edge(
        G,
        start_thr=ss["min_edge_weight"],
        target_max_edges=ss.get("target_max_edges", 200),
        cap=ss.get("min_edge_cap", 20),
    )
    if new_thr != ss["min_edge_weight"]:
        ss["min_edge_weight"] = new_thr
    st.caption(f"現在の閾値: {ss['min_edge_weight']}（エッジ数: {edge_cnt}）")
else:
    st.caption(f"現在の閾値: {ss['min_edge_weight']}（エッジ数: {_edge_count_at_threshold(G, ss['min_edge_weight'])}）")

st.subheader("単語間の関係性")

comms = render_pyvis_network(
    G,
    min_edge_weight=ss["min_edge_weight"],
    physics=ss["physics_on"],
    height_px=720,
    label_font_size=ss["label_font_size"],
    enable_clustering=True,
    focus_community=None
)

with st.expander("⚙️ ネットワーク表示オプション", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.slider("各レコードで使う上位語数", 10, 100, ss["top_k_per_doc"], 5, key="top_k_per_doc")
    with c2:
        st.slider("全体頻度の上位N語まで", 30, 200, ss["max_nodes_global"], 10, key="max_nodes_global")
    with c3:
        st.slider("エッジの最小共起回数", 1, 20, ss["min_edge_weight"], 1,
                  key="min_edge_weight", on_change=_mark_min_edge_touched)
    with c4:
        st.selectbox("重みの定義", ["binary", "mincnt"],
                     index=0 if ss["weight_mode"] == "binary" else 1, key="weight_mode")
    with c5:
        st.slider("ラベル文字サイズ", 12, 36, ss["label_font_size"], 1, key="label_font_size")

    st.toggle("レイアウトの物理シミュレーションを有効化", value=ss["physics_on"], key="physics_on")
    st.caption("※ オプション変更で自動的に再描画されます。")

# ===== ▼TOP10（各語の時系列 + 一緒に出てきたワード） =====
st.subheader("頻出単語 TOP10")
with st.expander("出現頻度の高い単語 TOP10", expanded=False):
    TOP_N = 10
    RELATED_K = 5
    threshold = ss["min_edge_weight"]

    top_terms = freq_counter.most_common(TOP_N)
    if not top_terms:
        st.write("なし")
    else:
        for rank, (term, cnt) in enumerate(top_terms, start=1):
            cooccur_rows = []
            if G.has_node(term):
                neighbors = []
                for nbr in G.neighbors(term):
                    w = int(G[term][nbr].get("weight", 1))
                    if w >= threshold:
                        neighbors.append((nbr, w))
                neighbors.sort(key=lambda x: x[1], reverse=True)
                cooccur_rows = neighbors[:RELATED_K]

            ts_rows = []
            for r in filtered:
                local = dict((t, int(c)) for t, c in r.get("top_terms", []))
                c_term = int(local.get(term, 0))
                if c_term <= 0:
                    continue
                dt = parse_date_from_chunk_head(r.get("chunk_id", ""))
                if dt is None:
                    continue
                ts_rows.append({"x": dt, "count": c_term})

            if ts_rows:
                df_ts = pd.DataFrame(ts_rows)
                ts_series = (
                    df_ts.groupby("x", dropna=True)["count"].sum().sort_index()
                )
                df_plot = ts_series.reset_index().rename(
                    columns={"x": "時点", "count": "出現数"}
                )
            else:
                df_plot = pd.DataFrame(columns=["時点", "出現数"])

            with st.expander(f"{rank}位：{term}（頻度：{cnt}回）", expanded=False):
                st.markdown("**該当語が出現した議会の時系列**")
                if not df_plot.empty:
                    df_m = df_plot.copy()
                    df_m["YYYY_MM"] = (
                        df_m["時点"]
                        .dt.to_period("M")
                        .astype(str)
                        .str.replace("-", ".", regex=False)
                    )
                    df_m = df_m.groupby("YYYY_MM", as_index=False)["出現数"].sum()
                    st.line_chart(df_m.set_index("YYYY_MM")["出現数"])
                else:
                    st.write("時系列データなし")

                st.markdown("**一緒に使われやすい単語（上位5件）**")
                if cooccur_rows:
                    st.dataframe(
                        {
                            "単語": [n for n, _ in cooccur_rows],
                            "頻度": [w for _, w in cooccur_rows],
                        },
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.write("なし（ネットワーク対象外 or しきい値が高い）")

# ===== ▼フッター =====
st.divider()
st.caption("""
⚠️ 本ページでは議会議事録などを形態素解析し、一般的な語句を除外したうえで頻出名詞を抽出して分析しています。  
⚠️ 処理軽減のため、各発言ごとに「その発言内で頻出した上位100語」のみを集計対象としています。  
🙌 本プロジェクトは個人により運営されています。ご支援いただける方はぜひこちらから：  
[💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")

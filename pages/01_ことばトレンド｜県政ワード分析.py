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

# 認証情報
# 1) st.secrets["aws"]["AWS_ACCESS_KEY"] / ["AWS_SECRET_KEY"]
# 2) 環境変数 AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
# 3) IAMロール（boto3のデフォルト）

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
    """
    S3上の .jsonl.zst をストリーム解凍し、TextIOWrapperでUTF-8として安全に行単位で読む。
    ・マルチバイト断片をTextIOWrapperが内部バッファで吸収するため decode エラーを回避
    ・巨大ファイルでもメモリ効率よく処理
    """
    s3 = make_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]  # StreamingBody (file-like)

    dctx = zstd.ZstdDecompressor()
    records = []

    # stream_reader(バイト) → TextIOWrapper(テキスト) で行ごとに読む
    with dctx.stream_reader(body) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="strict", newline="")
        for line in text_stream:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # もし末尾の不完全行などがあればスキップ（必要ならログに出す）
                # st.warning("不完全なJSON行をスキップしました")
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
    """
    records: フィルタ後のレコード群
    global_freq: 全体の頻度（freq_counter）
    top_k_per_doc: 各レコード内で上位何語を共起候補に使うか
    max_nodes_global: 全体頻度上位の語をこの数までに制限（描画軽量化）
    weight_mode: "binary"なら同時出現で+1, "mincnt"ならmin(cnt_i, cnt_j)を加算
    """
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
    # しきい値でスリム化
    H = nx.Graph()
    for n, attrs in G.nodes(data=True):
        H.add_node(n, **attrs)
    for u, v, attrs in G.edges(data=True):
        if int(attrs.get("weight", 1)) >= min_edge_weight:
            H.add_edge(u, v, **attrs)

    # 孤立ノードを削除
    if drop_isolates:
        H.remove_nodes_from(list(nx.isolates(H)))

    # ★追加: 最大連結成分のみ残す（任意）
    if keep_largest_component and H.number_of_nodes() > 0:
        comps = list(nx.connected_components(H))
        giant = max(comps, key=len)
        H = H.subgraph(giant).copy()

    # ---- コミュニティ抽出
    comm_map, comms = ({}, [])
    if enable_clustering and H.number_of_nodes() > 0:
        comm_map, comms = detect_communities(H)

    if focus_community is not None and enable_clustering and comms:
        keep = comms[focus_community] if focus_community < len(comms) else set()
        H = H.subgraph(keep).copy()
        if H.number_of_nodes() > 0:
            comm_map, comms = detect_communities(H)

    # ★ 初期レイアウト（重心→ローカル）を計算
    pos = layout_communities_with_warmstart(
        H, comms,
        cluster_k=2.0,     # まとまり間のバネ長（大きいほど離れる）
        local_k=0.5,       # まとまり内の密度
        cluster_scale=900, # まとまり間の距離スケール
        local_scale=550,   # まとまり内の広がりスケール
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

    # 初期座標を与え、physicsは動かす（fixed=False / physics=True）    
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
            physics=physics,     # トグル反映
            fixed=not physics    # 物理OFFなら固定
        )

    # エッジ追加
    for u, v, attrs in H.edges(data=True):
        w = int(attrs.get("weight", 1))
        net.add_edge(u, v, value=w, title=f"co-occur: {w}")
    # vis.js オプションもトグル反映
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
    """
    コミュニティ間の“超グラフ”を作る（ノード=コミュニティ、エッジ=相互接続の重み合計）
    """
    M = nx.Graph()
    for i, nodes in enumerate(comms):
        M.add_node(i, size=len(nodes))
    # コミュニティ間のエッジ重みを集計
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
    cluster_k: float = 2.0,     # クラスタ（重心）の“広がり”
    local_k: float = 0.5,       # コミュニティ内の“広がり”
    cluster_scale: float = 900, # クラスタ間のスケール
    local_scale: float = 550,   # コミュニティ内のスケール
    seed: int = 42
):
    """
    1) コミュニティ間のメタグラフを spring_layout
    2) 各コミュニティ内も spring_layout
    3) 重心にローカル配置をオフセット → {node: (x, y)} を返す
    """
    if not comms:
        return nx.spring_layout(H, k=local_k, weight="weight", seed=seed)

    # 1) 重心（コミュニティ）間レイアウト
    M = community_meta_graph(H, comms)
    pos_comm = nx.spring_layout(M, k=cluster_k, weight="weight", seed=seed)

    # 2) 各コミュニティ内レイアウト → 3) オフセット合成
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
    """重み付きGreedyでコミュニティ抽出し、{node: community_id}, [set(nodes)] を返す"""
    if G.number_of_nodes() == 0:
        return {}, []
    comms = list(greedy_modularity_communities(G, weight="weight"))
    comm_map = {}
    for i, s in enumerate(comms):
        for n in s:
            comm_map[n] = i
    return comm_map, comms

def summarize_communities(G: nx.Graph, comms):
    """各コミュニティの代表語（ノードsize降順TOP10）を返す"""
    summaries = []
    for i, nodes in enumerate(comms):
        rows = sorted(
            [(n, int(G.nodes[n].get("size", 1))) for n in nodes],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        summaries.append({"community": i, "top_terms": rows, "size": len(nodes)})
    return summaries

#ネットワークのグルーピング
def layout_by_community(H: nx.Graph, comms, *, intra_k=0.5, spacing=800, seed=42):
    """
    各コミュニティ内は spring_layout で詰めて配置し、
    コミュニティの“塊”同士は円周上に等間隔でオフセット。
    戻り値: {node: (x, y)}
    """
    if not comms:
        return nx.spring_layout(H, k=intra_k, weight="weight", seed=seed)

    pos = {}
    n_comm = len(comms)
    # コミュニティの中心（円周上）
    for i, nodes in enumerate(comms):
        angle = 2 * math.pi * i / n_comm
        cx = spacing * math.cos(angle)
        cy = spacing * math.sin(angle)

        # サブグラフを局所レイアウト（密に）
        sub = H.subgraph(nodes)
        local = nx.spring_layout(sub, k=intra_k, weight="weight", seed=seed)

        # 中心へオフセット
        for n, (x, y) in local.items():
            pos[n] = (x * 300 + cx, y * 300 + cy)  # 300 は塊の“直径”スケール
    return pos

def layout_by_community_grid(H: nx.Graph, comms, *,
                             cluster_spacing=1200,
                             subgraph_scale=500,
                             grid_cols=3,
                             seed=42):
    """
    コミュニティごとに spring_layout を計算し、グリッド配置で距離を空ける
    """
    if not comms:
        return nx.spring_layout(H, k=0.5, weight="weight", seed=seed)

    pos = {}
    for i, nodes in enumerate(comms):
        sub = H.subgraph(nodes)
        # クラスタ内レイアウト
        local = nx.spring_layout(sub, k=0.5, weight="weight", seed=seed)
        # 重心をグリッドに配置
        row, col = divmod(i, grid_cols)
        cx = col * cluster_spacing
        cy = row * cluster_spacing
        for n, (x, y) in local.items():
            pos[n] = (x * subgraph_scale + cx, y * subgraph_scale + cy)
    return pos

#エッジ数の自動調整
def _edge_count_at_threshold(G: nx.Graph, thr: int) -> int:
    return sum(1 for _, _, d in G.edges(data=True) if int(d.get("weight", 1)) >= thr)

def recommend_min_edge(G: nx.Graph, start_thr: int, *, target_max_edges: int, cap: int,
                       step_back_if_below: bool = True) -> tuple[int, int]:
    """
    しきい値を start_thr から1ずつ上げ、エッジ数が target_max_edges 以下になった時点で返す。
    その際、もし edges < target_max_edges * 0.6 なら 1段だけ戻して確定する。
    （戻すことで target_max_edges を超えても構わない、というポリシー）
    戻り値: (確定しきい値, そのときのエッジ本数)
    """
    thr = start_thr
    edges = _edge_count_at_threshold(G, thr)

    # 上限以下に収まるまで引き締め
    while edges > target_max_edges and thr < cap:
        thr += 1
        edges = _edge_count_at_threshold(G, thr)

    # 落ちすぎ判定：例）max=200 のとき、edges < 100 なら 1 段戻す
    if thr > start_thr and edges < target_max_edges * 0.6:
        prev_edges = _edge_count_at_threshold(G, thr - 1)
        return thr - 1, prev_edges

    return thr, edges

# コールバック：ユーザーがしきい値をいじったらフラグON
def _mark_min_edge_touched():
    st.session_state["min_edge_user_touched"] = True

# 各発言の年月日取得
def parse_date_from_chunk_head(chunk_id: str) -> pd.Timestamp | None:
    """
    例: '2024年03月26日_令和6年3月定例記者会見_001' → Timestamp('2024-03-26')
    """
    if not chunk_id:
        return None
    head = str(chunk_id).split("_", 1)[0]
    m = re.match(r"^\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", head)
    if not m:
        return None  # 想定外フォーマットなら None（あればログでもOK）
    y, mo, d = map(int, m.groups())
    try:
        return pd.Timestamp(y, mo, d)
    except Exception:
        return None

    # 2) 日なし（YYYY年MM月 → 月初日に寄せる）
    m = re.match(r"^\s*(\d{4})年\s*(\d{1,2})月\s*$", head)
    if m:
        y, mo = map(int, m.groups())
        try:
            return pd.Timestamp(y, mo, 1)
        except Exception:
            return None

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
/* MultiSelect の選択タグ（BaseWeb Tag） */
.stMultiSelect [data-baseweb="tag"]{
  background-color:#eef3f8 !important;  /* 薄い青グレー */
  color:#1f2937 !important;              /* 濃いグレー文字 */
  border:1px solid #d1d5db !important;   /* グレー罫線 */
}
.stMultiSelect [data-baseweb="tag"] [data-baseweb="tag-close-icon"]{
  color:#6b7280 !important;              /* 閉じる×の色も控えめに */
}
</style>
""", unsafe_allow_html=True)

with st.spinner("データを読み込んでいます…"):
    try:
        data = load_preagg_records(S3_BUCKET, S3_KEY)
    except Exception as e:
        st.error(f"サーバーからの読込に失敗しました。時間をおいて再度お試しください。")
        st.stop()

if not data:
    st.warning("事前集計データが空でした。")
    st.stop()

# フィルタ用のユニーク値収集
years          = sorted({rec.get("year") for rec in data if rec.get("year")})
meetings       = sorted({rec.get("meeting_name") for rec in data if rec.get("meeting_name")})
speakers       = sorted({rec.get("speaker") for rec in data if rec.get("speaker")})
speaker_roles  = sorted({rec.get("speaker_role") for rec in data if rec.get("speaker_role")})

# 表示順を「議員」「知事」「行政関係者」に固定（存在しないものはスキップ）
desired_order = ["議員", "知事","委員", "行政関係者"]
speaker_roles = [r for r in desired_order if r in speaker_roles] + [r for r in speaker_roles if r not in desired_order]

# --- ネットワーク系の初期値 ---
# 推奨デフォルト（きめ）
DEFAULT_TOPK = 30
DEFAULT_MAXN = 80
DEFAULT_MINEDGE = 2
DEFAULT_LABEL_FONT = 22
DEFAULT_WEIGHT_MODE = "binary"
DEFAULT_PHYSICS = True   

# session_state 初期化
ss = st.session_state
ss.setdefault("top_k_per_doc", DEFAULT_TOPK)
ss.setdefault("max_nodes_global", DEFAULT_MAXN)
ss.setdefault("min_edge_weight", DEFAULT_MINEDGE)
ss.setdefault("weight_mode", DEFAULT_WEIGHT_MODE)
ss.setdefault("label_font_size", DEFAULT_LABEL_FONT)
ss.setdefault("physics_on", DEFAULT_PHYSICS)

# 自動調整関連（必要な既定値）
ss.setdefault("min_edge_user_touched", False)
ss.setdefault("auto_min_edge", True)
ss.setdefault("target_max_edges", 200)  
ss.setdefault("min_edge_cap", 20)

# --- フィルタ UI（複数選択対応：合算できる）---
st.divider()
st.subheader("条件")

# “すべて” を実際の全選択に展開する共通関数（MultiSelect用）
def expand_all(selected, universe):
    if ("すべて" in selected) or (not selected):
        return set(universe)
    return set(selected)

# ---------- 1行目：発言者種別と 発言者 ----------

# 役職→発言者 の対応表の作成
speakers_by_role = defaultdict(set)
for r in data:
    role = r.get("speaker_role")
    sp   = r.get("speaker")
    if role and sp:
        speakers_by_role[role].add(sp)
        
c1, c2 = st.columns(2)

with c1:
    # 発言者区分：単一選択（selectbox）
    default_idx = speaker_roles.index("議員") if "議員" in speaker_roles else 0
    sel_role = st.selectbox("発言者区分", speaker_roles, index=default_idx, key="role_sb")

# 役職に応じた発言者候補
allowed_speakers = sorted(speakers_by_role.get(sel_role, set()))

with c2:
    # 発言者：単一選択（selectbox）。"すべて" も選べる
    speaker_options = ["すべて"] + allowed_speakers
    current = st.session_state.get("speaker_sb", "すべて")
    if current not in speaker_options:
        current = "すべて"
    sel_speaker = st.selectbox("発言者", speaker_options, index=speaker_options.index(current), key="speaker_sb")

# ---------- 2行目：発言年 と 会議名 ----------
c3, c4 = st.columns(2)
with c3:
    sel_years = st.multiselect("発言年", ["すべて"] + years, default=["すべて"])
with c4:
    sel_meet = st.multiselect("会議名", ["すべて"] + meetings, default=["すべて"])


# --------- セレクションを集合に展開 ----------
years_set = expand_all(sel_years, years)      # multiselect
meet_set  = expand_all(sel_meet, meetings)    # multiselect

#単一選択（selectbox）はスカラー
if sel_speaker == "すべて":
    speaker_set = set(allowed_speakers)
else:
    speaker_set = {sel_speaker}

# --------- フィルタ適用（qa_role条件を削除／speaker_roleは単一一致） ----------
filtered = [
    r for r in data
    if (r.get("year") in years_set)
    and (r.get("meeting_name") in meet_set)
    and (r.get("speaker") in speaker_set)
    and (r.get("speaker_role") == sel_role)
]
# ▼ フィルタ構成のシグネチャ（変更検知用）
filter_sig = (
    sel_role,
    tuple(sorted(speaker_set)),
    tuple(sorted(years_set)),
    tuple(sorted(meet_set)),
)

# フィルタが前回から変わっていれば、自動調整を再び有効化する初期状態へ戻す
if ss.get("last_filter_sig") != filter_sig:
    ss["last_filter_sig"] = filter_sig
    ss["min_edge_user_touched"] = False         # 手動フラグ解除
    ss["min_edge_weight"] = DEFAULT_MINEDGE     # デフォルトに戻す（=自動調整が走る条件）
    
# ===== ▼ワードクラウド 画像表示部 =====

st.divider()
st.subheader("頻出単語")

freq_counter, total_utterances = aggregate_terms(filtered)
draw_wordcloud(freq_counter)

#st.caption(f"該当キー数: {len(filtered)}")
#st.caption(f"合算対象の発言レコード数（utterances合計）: {total_utterances}")

# ===== ▼ネットワーク 表示部 =====


# --- まず現在値でネットワークを構築 ---
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

# --- ネットワークを描画（コミュニティ色分け）---
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
        # ← ここに on_change を追加
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

            # --- 共起（関連ワード） ---
            cooccur_rows = []
            if G.has_node(term):
                neighbors = []
                for nbr in G.neighbors(term):
                    w = int(G[term][nbr].get("weight", 1))
                    if w >= threshold:
                        neighbors.append((nbr, w))
                neighbors.sort(key=lambda x: x[1], reverse=True)
                cooccur_rows = neighbors[:RELATED_K]

            # --- 時系列（議会ごとの出現数） ---
            ts_rows = []
            for r in filtered:
                local = dict((t, int(c)) for t, c in r.get("top_terms", []))
                c_term = int(local.get(term, 0))
                if c_term <= 0:
                    continue
            
                dt = parse_date_from_chunk_head(r.get("chunk_id", ""))
                if dt is None:
                    continue  # 想定外を除外（必要なら st.warning など）
            
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
            
            # ---------- 表示 ----------
            with st.expander(f"{rank}位：{term}（頻度：{cnt}回）", expanded=False):

                # 時系列（上）
                st.markdown("**該当語が出現した議会の時系列**")
                if not df_plot.empty:
                    # 月次に集計（同月内を合算）
                    df_m = df_plot.copy()
                    df_m["YYYY_MM"] = (
                        df_m["時点"]
                        .dt.to_period("M")           # 月単位に丸め（Period）
                        .astype(str)                 # '2023-01' 形式
                        .str.replace("-", ".", regex=False)  # '2023.01' に
                    )
                    df_m = df_m.groupby("YYYY_MM", as_index=False)["出現数"].sum()
                
                    # Streamlitのネイティブ折れ線で見た目を維持（カテゴリ軸）
                    st.line_chart(df_m.set_index("YYYY_MM")["出現数"])
                else:
                    st.write("時系列データなし")
                

                # 一緒に出てきたワード（下）
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

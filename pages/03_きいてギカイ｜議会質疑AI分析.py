# -*- coding: utf-8 -*-
import streamlit as st
import json, random, re, boto3
from datetime import datetime
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from collections import defaultdict

st.set_page_config(page_title="きいてギカイ｜議会質疑AI分析", layout="wide", page_icon="📜")

# ====== ▼ 初期値設定 ========================================================
# 検索パラメータ
TOP_N_RETURN       = 10         
SIM_THRESHOLD      = 0.1        
TOPK_CANDIDATES    = 30         

#ChatGPTの設定
GPT_MODEL        = "gpt-4.1-mini"
GPT_TEMPERATURE  = 0.1
EMBED_MODEL      = "text-embedding-3-small"  

#AWSの設定
AWS_REGION       = "us-west-2"
OUTPUT_PREFIX    = "council_chunk_jsonl_ui/"  
#OUTPUT_PREFIX    = "council_chunk_jsonl/"  
AWS_ACCESS_KEY_S = st.secrets["AWS-KEY"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY_S = st.secrets["AWS-KEY"]["AWS_SECRET_KEY"]
DATA_BUCKET_NAME = st.secrets["AWS-KEY"]["DATA_BUCKET_NAME"]
S3_INDEX_ARN     = st.secrets["AWS-KEY"]["VECTOR_INDEX_ARN_COUNCIL"]
# ====== ▲ 初期値設定 ========================================================

# ========================= セッション初期化 =========================
for key in ["agreed", "query", "send_now", "last_answer", "last_matches", "is_generating",
            "input", "input_value", "suggestions_sampled", "qa_pairs", "clarified", "clarify_active"]:
    if key not in st.session_state:
        if key in ["query", "last_answer", "input", "input_value"]:
            st.session_state[key] = ""
        elif key == "suggestions_sampled":
            st.session_state[key] = []
        elif key in ["last_matches", "qa_pairs"]:
            st.session_state[key] = []
        elif key == "clarify_active":
            st.session_state[key] = False
        else:
            st.session_state[key] = False

# ========================= ユーティリティ =========================
def load_prompt(filename: str, default_text: str = "") -> str:
    try:
        with open("prompts/" + filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default_text

def _to_similarity(distance: float) -> float:
    """cosine距離 → 類似度(1 - distance)"""
    try:
        d = float(distance)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, 1.0 - d))

def _base_from_chunk_id(chunk_id: str) -> str:
    """'xxxx_000' → 'xxxx'"""
    return re.sub(r"_[0-9]{1,3}$", "", chunk_id or "")

# ========================= クライアント =========================
client_oai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def _boto_s3():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_S,
        aws_secret_access_key=AWS_SECRET_KEY_S,
        region_name=AWS_REGION,
    )

def _boto_s3vectors():
    return boto3.client(
        "s3vectors",
        aws_access_key_id=AWS_ACCESS_KEY_S,
        aws_secret_access_key=AWS_SECRET_KEY_S,
        region_name=AWS_REGION,
    )

# ========================= JSONLアクセス（S3 Selectなし） =========================
_jsonl_cache = {}

def _load_jsonl_lines(s3_client, base_name: str):
    """
    OUTPUT_PREFIX/{base}.jsonl を取得して全行をパースし、メモリキャッシュする。
    """
    key = f"{OUTPUT_PREFIX}{base_name}.jsonl"
    if key in _jsonl_cache:
        return _jsonl_cache[key]

    try:
        body = s3_client.get_object(Bucket=DATA_BUCKET_NAME, Key=key)["Body"].read().decode("utf-8")
    except Exception:
        _jsonl_cache[key] = []
        return _jsonl_cache[key]

    rows: List[Dict[str, Any]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            rows.append(obj)
        except Exception:
            continue

    _jsonl_cache[key] = rows
    return rows

def _get_pair_records_from_file(rows, pair_id):
    """
    事前に読み込んだrowsから pair_id に一致する行を返す。Q→A→Nの順に整列。
    """
    pid_int = int(pair_id)
    pid_str = str(pid_int)
    recs = [r for r in rows if (r.get("pair_id") == pid_int or str(r.get("pair_id")) == pid_str)]
    order = {"Q": 0, "A": 1, "N": 2}
    recs.sort(key=lambda r: order.get(r.get("qa_role"), 9))
    return recs

# ========================= ベクター検索（S3 Vectors） =========================
def _query_s3vectors(query_text):
    emb = client_oai.embeddings.create(model=EMBED_MODEL, input=query_text)
    qvec = [float(x) for x in emb.data[0].embedding]

    s3v = _boto_s3vectors()
    res = s3v.query_vectors(
        indexArn=S3_INDEX_ARN,
        queryVector={"float32": qvec},
        topK=TOPK_CANDIDATES,
        returnMetadata=True,
        returnDistance=True,
    )
    matches = res.get("vectors", []) or []

    out = []
    for m in matches:
        distance = float(m.get("distance", 0.0))
        score = max(0.0, min(1.0, 1.0 - distance))
        if score < SIM_THRESHOLD:
            continue

        md = m.get("metadata") or {}
        out.append({
            "score": score,
            "distance": distance,
            "key": m.get("key") or m.get("id"),
            "source_file": md.get("source_file") or md.get("source_id") or md.get("source") or "",
            "chunk_id": md.get("chunk_id") or md.get("chank_id") or (m.get("key") or m.get("id")),
            "pair_id": int(float(md["pair_id"])) if md.get("pair_id") is not None else None,
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out

# ========================= Google Sheets ログ =========================
def get_gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gsheets_service_account"], scope)
    return gspread.authorize(creds)

def log_to_gsheet(question, answer):
    try:
        client_gs = get_gspread_client()
        sheet = client_gs.open_by_key(st.secrets["kiite-gikai"]["GOOGLE_GIKAI_LOG_SHEET_ID"]).worksheet("logs")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([now, question, answer])
    except Exception as e:
        st.warning(f"⚠️ ログ記録に失敗しました")

# ========================= clarify機能 =========================
def clarify_query(user_query):
    clarify_prompt = load_prompt("gikai_clarify_prompt.txt")
    messages = [
        {"role": "system", "content": clarify_prompt},
        {"role": "user", "content": f"【質問】{user_query}"},
    ]
    try:
        response = client_oai.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=GPT_TEMPERATURE
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"Clarifyエラー")
        return {"ambiguous": False, "reason": "", "rewritten_query": ""}

# ========================= 検索〜要約 本体 =========================
def search_s3vector_and_respond(query):
    """
    - S3Vectorsで候補を取得（score=1-distanceでfilter）
    - (base, pair_id) のユニーク集合を作る
    - 各baseに対して一度だけ JSONL をロード → pair_id一致行を抽出 → Q/A/Nに振り分け
    - Q/Aが揃ったペアを要約 → 全体要約
    """
    s3_client = _boto_s3()

    # 1) ベクトル検索
    try:
        hits = _query_s3vectors(query)
    except Exception as e:
        return {"matches": [], "summary": f"🔍 検索エラーが発生しました", "qa_pairs": []}

    if not hits:
        return {"matches": [], "summary": "🔍 類似度の高い発言は見つかりませんでした。", "qa_pairs": []}

    # 2) 画面用の断片（ペア化できない時のフォールバック表示用）
    top_hits = hits[:TOP_N_RETURN]
    top_matches_for_ui = []
    for h in top_hits:
        base = _base_from_chunk_id(h.get("chunk_id", ""))
        top_matches_for_ui.append({
            "text": "",  
            "topic": "未分類",
            "source_file": h.get("source_file") or f"{base}.txt",
            "date": None,
            "type": None,
            "score": float(h.get("score", 0.0)),
            "chunk_id": h.get("chunk_id"),
            "speaker": "",
            "speaker_role": "",
            "source_id": h.get("source_file") or "",
            "source_index": "s3vectors",
        })

    # 3) (base, pair_id) でユニーク化してQ/A抽出
    #    pair_id が None のヒットはペア組成対象から除外（UI断片のみ）
    unique_pairs = []
    for h in top_hits:
        pid = h.get("pair_id")
        if pid is None:
            continue
        base = _base_from_chunk_id(h.get("chunk_id", ""))
        key = (base, int(pid))
        if key not in unique_pairs:
            unique_pairs.append(key)
                    
    need = defaultdict(list) 
    for base, pid in unique_pairs:
        need[base].append(pid)

    pair_matches = []
    for base, pid_list in need.items():
        rows = _load_jsonl_lines_select_jsonl(s3_client, base, pid_list)  
        if not rows:
            continue

        # 各 pid ごとに Q/A 抜き出し
        for pid in pid_list:
            recs = [r for r in rows if int(r.get("pair_id", -1)) == int(pid)]
            if not recs:
                continue
            q = [r for r in recs if r.get("qa_role") == "Q"]
            a = [r for r in recs if r.get("qa_role") == "A"]
            if not q and not a:
                continue

            first_cid = ((q and q[0].get("chunk_id")) or (a and a[0].get("chunk_id")) or "")
            meeting_name = first_cid.split("_")[0] if first_cid else "（会議名不明）"

            def _enrich(rec): 
                return {**rec, "source_file": f"{base}.txt"}

            pair_matches.append({
                "pair_id": int(pid),
                "source_file": f"{base}.txt",
                "meeting_name": meeting_name,
                "Q": [_enrich(r) for r in q],
                "A": [_enrich(r) for r in a],
            })


    # 4) サマリ生成

    guard = f"""
    ---
    【追加制約】
    - 以降の要約は、ユーザーの質問「{query}」に直接関連する情報のみを対象にしてください。
    - 関連しない内容が含まれている場合は無視してください。
    """
            
    gikai_pair_prompt = load_prompt("gikai_pair_summary.txt") + "\n" + guard
    summary_overall_prompt = load_prompt("gikai_summary_overall.txt") + "\n" + guard

    summary_per_pair = []
    for pair in pair_matches:
        q_blocks = []
        for q in pair.get("Q", []):
            role = q.get("speaker_role", "")
            speaker = q.get("speaker", "")
            text = q.get("text", "")
            if text:
                q_blocks.append(f"【質問】{role} {speaker}：{text}")

        a_blocks = []
        for a in pair.get("A", []):
            role = a.get("speaker_role", "")
            speaker = a.get("speaker", "")
            text = a.get("text", "")
            if text:
                a_blocks.append(f"【答弁】{role} {speaker}：{text}")

        qa_context = "\n\n".join(q_blocks + a_blocks) or "(該当文脈なし)"

        try:
            resp = client_oai.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": gikai_pair_prompt},
                    {"role": "user", "content": qa_context}
                ],
                temperature=GPT_TEMPERATURE
            )
            summary = resp.choices[0].message.content.strip()
        except Exception as e:
            summary = f"⚠️ 要約失敗"

        pair["summary"] = summary
        summary_per_pair.append(summary)

    if summary_per_pair:
        try:
            context = "\n\n".join([f"{i+1}件目：{s}" for i, s in enumerate(summary_per_pair)])
            resp = client_oai.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": summary_overall_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=GPT_TEMPERATURE
            )
            summary_overall = resp.choices[0].message.content.strip()
        except Exception as e:
            summary_overall = f"⚠️ 全体サマリ生成失敗"
    else:
        # ペアが一つも組めなかった場合は断片ベースの注意喚起
        summary_overall = "⚠️ Q/Aペアを組める一致は見つかりませんでした。参考までに関連断片を表示します。"

    return {
                "matches": top_hits[:TOP_N_RETURN] and top_matches_for_ui or [],
                "summary": summary_overall,
                "qa_pairs": pair_matches
    }


# ========================= S3 Selectでの対象データ抽出 =========================
from collections import defaultdict
def _load_jsonl_lines_select_jsonl(s3_client, base_name: str, pair_ids: list[int]):
    """
    S3 Selectで OUTPUT_PREFIX/{base}.jsonl（非圧縮）から
    指定 pair_id の行だけを抽出して返す。
    失敗時は従来の全件GET→ローカル抽出にフォールバック。
    """
    import json
    from botocore.exceptions import ClientError

    key_raw = f"{OUTPUT_PREFIX}{base_name}.jsonl"
    ids = sorted({int(x) for x in pair_ids if x is not None})
    if not ids:
        return []

    # ---- 1) S3 Select（非圧縮JSON Lines） ----
    try:
        expr_ids = ",".join(str(i) for i in ids)
        resp = s3_client.select_object_content(
            Bucket=DATA_BUCKET_NAME,
            Key=key_raw,
            ExpressionType='SQL',
            Expression=(
                "SELECT s.pair_id, s.qa_role, s.chunk_id, s.text, "
                "       s.speaker, s.speaker_role, s.source_file "
                f"FROM S3Object s WHERE cast(s.pair_id as int) IN ({expr_ids})"
            ),
            InputSerialization={  # 非圧縮なので CompressionType は指定しない
                "JSON": {"Type": "LINES"}
            },
            OutputSerialization={"JSON": {}},
        )

        out = []
        for event in resp["Payload"]:
            if "Records" in event:
                payload = event["Records"]["Payload"].decode("utf-8")
                for line in payload.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except:
                        pass
        return out

    except ClientError:
        # Keyが無い/Select不可など → フォールバック
        pass

    # ---- 2) フォールバック：従来の全件GET→ローカル抽出 ----
    try:
        body = s3_client.get_object(Bucket=DATA_BUCKET_NAME, Key=key_raw)["Body"].read().decode("utf-8")
        rows, ids_set = [], set(ids)
        for line in body.splitlines():
            if not line:
                continue
            try:
                obj = json.loads(line)
                if int(obj.get("pair_id", -1)) in ids_set:
                    rows.append(obj)
            except:
                pass
        return rows
    except Exception:
        return []


# ========================= UI =========================
st.title("📜 きいてギカイ｜議会質疑AI分析")

# 同意画面
if not st.session_state.agreed:
    st.markdown("""
    ### ご利用にあたってのご案内
    - このチャットでは、山口市議会の一般質問・質疑の議事録をもとに、議会でどんな議論が行われているかを知ることができます。  
    - チャット内容は記録されます。内容の記録に同意された方のみ、チャットをご利用ください。
    - **個人情報（氏名・住所・連絡先など）の入力は行わないでください。**  
    """)
    st.warning("このチャットを利用するには、以下の内容に同意いただく必要があります。")
    if st.button("✅ 同意してチャットをはじめる"):
        st.session_state.agreed = True
        st.rerun()
    st.stop()

# 入力欄
st.markdown("---\n\n#### 💬 質問してみよう")
st.text_input(
    label="",
    key="input",
    value=st.session_state.input_value,
    placeholder="例：学校の給食費無償化についてどのような議論が行われていますか",
    on_change=lambda: st.session_state.update(send_now=True)
)

# サジェスト
if not st.session_state.get("clarify_active", False):
    suggestions_master = [
        "公共施設の統廃合について気になります",
        "行政のDX化は進んでいますか",
        "観光・インバウンド対応について教えて"
    ]
    if not st.session_state.suggestions_sampled:
        st.session_state.suggestions_sampled = random.sample(suggestions_master, k=3)

    cols = st.columns(3)
    for i, s in enumerate(st.session_state.suggestions_sampled):
        if cols[i].button(f" {s}", key=f"sugg_{s}"):
            st.session_state.input_value = s
            st.session_state.query = s
            st.session_state.send_now = False
            st.session_state.is_generating = True
            st.session_state.clarify_active = False
            with st.spinner(f"⏳ 「{s}」に回答中... 少々お待ちください"):
                results = search_s3vector_and_respond(s)
                st.session_state.last_answer = results["summary"]
                st.session_state.last_matches = results["matches"]
                st.session_state.qa_pairs = results["qa_pairs"]
                log_to_gsheet(s, results["summary"])
            st.session_state.is_generating = False
            st.rerun()

# Clarify
if st.session_state.input and not st.session_state.get("clarified", False):
    clarify_result = clarify_query(st.session_state.input)
    if clarify_result["ambiguous"] and clarify_result["rewritten_query"]:
        st.session_state.clarify_active = True
        st.info(f"👇 より正確な検索のため「{clarify_result['reason']}」ことをお勧めします。\n\n**→ {clarify_result['rewritten_query']}**")

        col1, col2 = st.columns(2)
        if col1.button("🔁 置き換えて検索"):
            st.session_state.input_value = clarify_result["rewritten_query"]
            st.session_state.clarified = True
            st.session_state.clarify_active = False
            st.session_state.send_now = True
            st.rerun()
        if col2.button("🔜 入力文のままで検索"):
            st.session_state.clarified = True
            st.session_state.clarify_active = False
            st.session_state.send_now = True
            st.rerun()
        st.stop()
    else:
        st.session_state.clarified = True

# Enter送信
if st.session_state.input and st.session_state.send_now:
    st.session_state.send_now = False
    st.session_state.is_generating = True
    with st.spinner(f"⏳ 「{st.session_state.input}」に回答中... 少々お待ちください"):
        results = search_s3vector_and_respond(st.session_state.input)
        st.session_state.last_answer = results["summary"]
        st.session_state.last_matches = results["matches"]
        st.session_state.qa_pairs = results["qa_pairs"]
        log_to_gsheet(st.session_state.input, results["summary"])
    st.session_state.input_value = ""
    st.session_state.is_generating = False

# 回答表示
if st.session_state.is_generating:
    st.info("⏳ 回答中... 少々お待ちください")

elif st.session_state.last_answer and st.session_state.qa_pairs:
    st.markdown("#### 💡 議会質問のまとめ")
    st.success(st.session_state.last_answer)

    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)

    # ✅ Q/Aペアを会議名ごとに分類（pairに埋め込んだ meeting_name を最優先）
    for p in st.session_state.qa_pairs:
        cid = (
            p.get("chunk_id")
            or (p.get("Q") and p["Q"][0].get("chunk_id"))
            or (p.get("A") and p["A"][0].get("chunk_id"))
            or ""
        )
        meeting_name = p.get("meeting_name") or (cid.split("_")[0] if cid else "（会議名不明）")
        meeting_name = meeting_name.split("：")[-1]
        groups[meeting_name].append(p)

    # 会議ごとにタブ化（Q/A件数だけバッジ表示）
    st.markdown("#### 💡 関連発言の詳細")
    meeting_names = sorted(groups.keys(), reverse=True)
    tabs = st.tabs([f"{mn}（ {len(groups[mn])}）" for mn in meeting_names])

    for tab, mn in zip(tabs, meeting_names):
        with tab:
            # Q/Aリスト
            for i, pair in enumerate(groups[mn], start=1):
                summary = (pair.get("summary") or "").strip()
                if not summary:
                    continue
                st.markdown(f"##### {i}. {summary}")
                for q in pair.get("Q", []):
                    with st.expander(f"🟢【質問】{q.get('speaker_role','')} {q.get('speaker','')}（{q.get('source_file','').replace('.txt','')}）"):
                        st.markdown(q.get("text", ""))
                for a in pair.get("A", []):
                    with st.expander(f"🔵【答弁】{a.get('speaker_role','')} {a.get('speaker','')}（{a.get('source_file','').replace('.txt','')}）"):
                        st.markdown(a.get("text", ""))
                st.divider()

            # 公式リンク
            st.markdown(
                """
                <div style="
                    background-color:#f5f5f5;
                    border:1px solid #ddd;
                    border-radius:6px;
                    padding:0.8em 1em;
                    margin-top:0.8em;
                    margin-bottom:0.8em;">
                    🔗 公式情報はこちらからご覧いただけます<br>
                    ・ <a href="https://www.city.yamaguchi.yamaguchi.dbsr.jp/index.php/" target="_blank">
                        山口市議会 議事録（公式HP）
                      </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

elif st.session_state.send_now or st.session_state.input.strip() or st.session_state.query:
    st.warning("⚠️ 情報が見つかりませんでした。")            

# フッター
st.divider()
st.caption("""
⚠️ 回答は生成AIによるものであり、正確性を保証するものではありません。  
🙌 本プロジェクトは個人により運営されています。ご支援いただける方はぜひこちらから：  
[💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")

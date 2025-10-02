import streamlit as st
import json, random, io, re, boto3
import numpy as np
from datetime import datetime
from openai import OpenAI
from oauth2client.service_account import ServiceAccountCredentials
import gspread

st.set_page_config(page_title="きいてミライ｜議会発言AI分析", layout="wide", page_icon="🏛️")

def render_items(items):
    if not items:
        st.info("該当なし")
        return

    for m in items:
        source_file = m.get("source_file", "").replace(".txt", "")
        date = m.get("date")
        topic = m.get("topic", "未分類")
        header = m.get("title") or m.get("snippet") or source_file or "関連発言"
        source = f"""<span style="font-size:0.9em; color:gray;">{source_file}</span>"""

        with st.expander(f"{topic}" if date else header, expanded=False):
            st.markdown(m.get("text", "") or "_(text空)_")
            st.markdown(source, unsafe_allow_html=True)

        
# ====== ▼ 初期値設定 ========================================================

#検索結果の設定
TOP_N_RETURN       = 10         # 最終的に返す件数
SIM_THRESHOLD      = 0.1        # 類似度のしきい値（0.0～1.0）
TOPK_CANDIDATES    = 30         # S3Vectorsから一旦取り寄せる候補数（多めに）

#ChatGPTの設定
GPT_MODEL = "gpt-4.1-mini"
GPT_TEMPERATURE = 0.1
EMBED_MODEL         = "text-embedding-3-small"  

#AWSの設定
AWS_REGION          = "us-west-2"
OUTPUT_PREFIX       = "pref_yamaguchi/chunks/" 
SCORE_THRESHOLD     = 0.0
AWS_ACCESS_KEY_S    = st.secrets["AWS-KEY"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY_S    = st.secrets["AWS-KEY"]["AWS_SECRET_KEY"]     
DATA_BUCKET_NAME    = st.secrets["AWS-KEY"]["DATA_BUCKET_NAME"]        
S3_INDEX_ARN        = st.secrets["AWS-KEY"]["VECTOR_INDEX_ARN"]
# ====== ▲ 初期値設定 ========================================================

for key in ["agreed", "query", "send_now", "last_answer", "last_matches", "is_generating", "input", "input_value", "clarified", "clarify_active", "suggestions_sampled"]:
    if key not in st.session_state:
        if key in ["query", "last_answer", "input", "input_value"]:
            st.session_state[key] = ""
        elif key == "suggestions_sampled":
            st.session_state[key] = []
        else:
            st.session_state[key] = False

if not st.session_state.agreed:
    st.title("🏛️きいてミライ｜議会発言AI分析")
    st.markdown("""
    ### ご利用にあたってのご案内

    - このチャットでは、議会での過去の発言をもとに、各発言者の見解を知ることができます。 
    - チャット内容は記録されます。内容の記録に同意された方のみ、チャットをご利用ください。 
    - **個人情報（氏名・住所・連絡先など）の入力は行わないでください。**  
    """)
    st.warning("このチャットを利用するには、以下の内容に同意いただく必要があります。")
    if st.button("✅ 同意してチャットをはじめる"):
        st.session_state.agreed = True
        st.rerun()
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gsheets_service_account"], scope)
    return gspread.authorize(creds)

def log_to_gsheet(question, answer):
    client_gs = get_gspread_client()
    sheet = client_gs.open_by_key(st.secrets["GOOGLE"]["PREFYAM_LOG_SHEET_ID"]).worksheet("logs")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([now, question, answer])

def load_prompt(name):
    with open(f"prompts/{name}", "r", encoding="utf-8") as f:
        return f.read()

def clarify_query(user_query):
    clarify_prompt = load_prompt("mirai_clarify_prompt.txt")
    messages = [
        {"role": "system", "content": clarify_prompt},
        {"role": "user", "content": f"【質問】{user_query}"}
    ]
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=GPT_TEMPERATURE
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        st.error(f"Clarifyエラー")
        return {"ambiguous": False, "reason": "", "rewritten_query": ""}

# ========= ▼ ここからS3 Vectorsでの検索を実装 ===============================

def _to_similarity(distance: float) -> float:
    """distance（cosine想定）→ 類似度スコア 1 - distance"""
    try:
        d = float(distance)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, 1.0 - d))

def _fetch_original_chunk_for_search(s3_client, source_file: str, chunk_id: str) -> tuple[dict | None, str]:
    """
    元jsonlの該当chunk(dict) と、参照したS3キー(文字列)を返す。
    見つからなかった場合は (None, key) を返す。
    """
    base = source_file
    key  = f"{OUTPUT_PREFIX}{base}.jsonl"
    try:
        body = s3_client.get_object(Bucket=DATA_BUCKET_NAME, Key=key)["Body"].read().decode("utf-8")
    except Exception:
        return None, key

    for line in body.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("chunk_id") == chunk_id:
            return obj, key
    return None, key

def _query_s3vectors(query_text: str, top_k_: int, score_threshold: float):
    # OpenAI埋め込み
    oai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    emb = oai.embeddings.create(model=EMBED_MODEL, input=query_text)
    qvec = [float(x) for x in emb.data[0].embedding]

    # S3 / S3Vectors クライアント
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_S,
        aws_secret_access_key=AWS_SECRET_KEY_S,
        region_name=AWS_REGION,
    )
    s3v_client = boto3.client(
        "s3vectors",
        aws_access_key_id=AWS_ACCESS_KEY_S,
        aws_secret_access_key=AWS_SECRET_KEY_S,
        region_name=AWS_REGION,
    )

    # 多めに取得
    res = s3v_client.query_vectors(
        indexArn=S3_INDEX_ARN,
        queryVector={"float32": qvec},
        topK=max(TOPK_CANDIDATES, top_k_),
        returnMetadata=True,
        returnDistance=True,
    )
    matches = res.get("vectors", []) or []

    out = []
    for m in matches:
        key       = m.get("key") or m.get("id")
        distance  = float(m.get("distance", 0.0))
        score     = _to_similarity(distance)
        if score < score_threshold:
            continue

        md        = m.get("metadata") or {}
        source    = md.get("source_file")
        chunk_id  = md.get("chunk_id") or key

        original, jsonl_key = _fetch_original_chunk_for_search(s3_client, source, chunk_id)

        out.append({
            "score": score,
            "distance": distance,
            "key": key,
            "source_file": source,
            "chunk_id": chunk_id,
            "original": original
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out

# 発言録を検索
def search_s3vector_and_respond(query):
    try:
        hits = _query_s3vectors(
            query_text=query,
            top_k_=TOP_N_RETURN,            
            score_threshold=SIM_THRESHOLD   
        )
    except Exception as e:
        return {"matches": [], "summary": f"🔍 検索エラーが発生しました"}

    if not hits:
        return {"matches": [], "summary": "🔍 類似度の高い発言は見つかりませんでした。"}

    # ✅ 類似度で降順 → 上位10件だけ
    top_hits = hits[:TOP_N_RETURN]

    # 返却形の整形（UIには「スコア＝類似度」を出すよう変更）
    top_matches = []
    for h in top_hits:
        o = h.get("original") or {}
        top_matches.append({
            "text": o.get("text") or "",
            "topic": o.get("topic") or "未分類",
            "source_file": h.get("source_file") or o.get("source_file") or "",
            "date": o.get("date"),
            "type": o.get("type"),
            "score": float(h.get("score", 0.0)),  # ← 類似度（高いほど良い）
            "chunk_id": h.get("chunk_id"),
            "source_index": "s3vectors",
        })

    # 🧠 上位10件の本文をまとめて要約
    try:
        combined_text = "\n\n".join(m["text"] for m in top_matches if m.get("text"))

        base_prompt = load_prompt("mirai_summary.txt")
        guard = f"""
        ---
        【追加制約】
        - 以降の要約は、ユーザーの質問「{query}」に直接関連する情報のみを対象にしてください。
        - 関連しない内容が含まれている場合は無視してください。
        """
        summary_prompt = base_prompt + "\n" + guard
        
        messages = [
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": combined_text if combined_text else "(該当する文脈が見つかりませんでした)"},
        ]
        resp = client.chat.completions.create(model=GPT_MODEL, messages=messages, temperature=GPT_TEMPERATURE)
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        summary = f"⚠️ 全体サマリ生成失敗"

    return {"matches": top_matches, "summary": summary}

    # 日付の降順に並べる:
    # items = sorted(items, key=lambda x: str(x.get("date", "")), reverse=True)

    for m in items:
        source_file = m.get("source_file", "").replace(".txt", "")
        date = m.get("date")
        topic = m.get("topic", "未分類")
        # expander の見出し（タイトル優先→スニペット→source_file）
        header = m.get("title") or m.get("snippet") or source_file or "関連発言"

        # 出典情報
        source = f"""<span style="font-size:0.9em; color:gray;">{source_file}</span>"""

        # タブ側で type を確定済みなので、expander 見出しは内容に集中
        with st.expander(f"{topic}" if date else header, expanded=False):
            st.markdown(m.get("text", ""))
            st.markdown(source, unsafe_allow_html=True)

st.title("🏛️ きいてミライ｜議会発言AI分析")

# --- チャット欄（送信ボタンなし・Enter送信） ---
st.markdown("---\n\n#### 💬 質問してみよう")
st.text_input(
    label="",
    key="input",
    value=st.session_state.input_value,
    placeholder="例：インバウンド観光に対する取り組みは？",
    on_change=lambda: st.session_state.update(send_now=True)
)

# --- サジェスト ---
if not st.session_state.get("clarify_active", False):
    suggestions_master = [
        "防災に関する発言はありますか？",
        "子育て支援の方針について教えて",
        "地域活性化に向けた取り組みは？"
    ]
    if not st.session_state.suggestions_sampled:
        st.session_state.suggestions_sampled = random.sample(suggestions_master, k=3)

    cols = st.columns(3)
    for i, s in enumerate(st.session_state.suggestions_sampled):
        if cols[i].button(s, key=f"sugg_{s}"):
            st.session_state.input_value = s
            st.session_state.query = s
            st.session_state.send_now = True
            st.session_state.is_generating = True
            st.session_state.clarify_active = False
            with st.spinner(f"⏳ 「{s}」に回答中... 少々お待ちください"):
                results = search_s3vector_and_respond(s)
                st.session_state.last_answer = results["summary"]
                st.session_state.last_matches = results["matches"]

                # ログ記録
                try:
                    log_to_gsheet(s, results["summary"])
                except Exception as e:
                    st.warning(f"⚠️ ログ記録に失敗しました")

            st.session_state.is_generating = False
            st.rerun()

# Clarifyプロンプトの確認
if st.session_state.input and not st.session_state.get("clarified", False):
    clarify_result = clarify_query(st.session_state.input)

    if clarify_result["ambiguous"] and clarify_result["rewritten_query"]:
        st.session_state.clarify_active = True
        st.info(f"👇 より正確な検索のため「{clarify_result['reason']}」ことをお勧めします。例えば、以下の質問に置き換えるのはいかがでしょうか？\n\n**→ {clarify_result['rewritten_query']}**")

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
        # 🟨 あいまいでない or 明確な修正提案がない場合も、Clarifyは終了
        st.session_state.clarified = True

# --- 送信処理（Enter or サジェスト選択時） ---
if st.session_state.input and st.session_state.send_now:
    st.session_state.send_now = False
    st.session_state.is_generating = True
    with st.spinner(f"⏳ 「{st.session_state.input}」に回答中... 少々お待ちください"):
        results = search_s3vector_and_respond(st.session_state.input)
        st.session_state.last_answer = results["summary"]
        st.session_state.last_matches = results["matches"]

        # ログ記録
        try:
            log_to_gsheet(st.session_state.input, results["summary"])
        except Exception as e:
            st.warning(f"⚠️ ログ記録に失敗しました")

    st.session_state.input_value = ""
    st.session_state.is_generating = False

if st.session_state.is_generating:
    st.info("⏳ 回答中... 少々お待ちください")
elif st.session_state.last_answer:
    st.divider()
    st.caption("入力された質問に対して、AIが類似度が高いと判断した上位10件の発言をもとに回答します")
    st.markdown("#### 💡 発言のまとめ")
    st.success(st.session_state.last_answer)  #  サマリ本文の出力位置
    
    st.subheader("関連発言の詳細")
    

    
    # 1) まず分類
    press_items = []    # 定例会見
    council_items = []  # 市政概況報告 

    for m in st.session_state.last_matches:
        source_file_raw = m.get("source_file", "")
        source_file = source_file_raw.replace(".txt", "")
        date = m.get("date")
        topic = m.get("topic", "未分類")
        type = "定例会見" if "会見" in source_file else "議会 市政概況報告"

        # 後段の描画で使いやすいよう整形して append
        enriched = {
            **m,
            "type": type,
            "topic": topic,
            "source_file": source_file,
            "date": date,
        }
        (press_items if type == "定例会見" else council_items).append(enriched)

    # 2) タブで分けて表示
    tabs = st.tabs([f"定例会見（{len(press_items)}）", f"議会 市政概況報告（{len(council_items)}）"])
    
    with tabs[0]:
        render_items(press_items)
        # 一次ソースを明示
        st.markdown(
            """
            <div style="
                background-color:#f5f5f5;
                border:1px solid #ddd;
                border-radius:6px;
                padding:0.8em 1em;
                margin-top:0.8em;
                margin-bottom:0.8em;
                ">
                🔗 公式情報はこちらからご覧いただけます<br>
                ・ <a href="https://ssp.kaigiroku.net/tenant/prefyamaguchi/pg/index.html" target="_blank">
                    山口県議会議事録
                  </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
            
    with tabs[1]:
        render_items(council_items)
        st.markdown(
            """
            <div style="
                background-color:#f5f5f5;
                border:1px solid #ddd;
                border-radius:6px;
                padding:0.8em 1em;
                margin-top:0.8em;
                margin-bottom:0.8em;
                ">
                🔗 公式情報はこちらからご覧いただけます<br>
                ・ <a href="https://ssp.kaigiroku.net/tenant/prefyamaguchi/pg/index.html" target="_blank">
                    山口県議会議事録
                  </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.caption("""
⚠️ 回答は生成AIによるものであり、正確性を保証するものではありません。  
🙌 本プロジェクトは個人により運営されています。ご支援いただける方はぜひこちらから：  
[💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")

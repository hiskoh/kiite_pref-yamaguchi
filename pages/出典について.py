# -*- coding: utf-8 -*-
import boto3, botocore
import streamlit as st

# ===== 設定 =====
S3_BUCKET      = st.secrets["AWS-KEY"]["DATA_BUCKET_NAME"]
AWS_REGION     = "us-west-2"  # 初期値
AWS_ACCESS_KEY = st.secrets["AWS-KEY"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS-KEY"]["AWS_SECRET_KEY"]

PREFIX_MAYOR   = "mayor_chunk_jsonl/"
PREFIX_COUNCIL = "council_chunk_jsonl_ui/"

st.set_page_config(page_title="出典一覧", layout="wide")
st.title("📂 出典一覧")

def make_client(region: str):
    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

# --- バケットの実リージョン取得（権限なければ初期値で続行） ---
try:
    loc = make_client(AWS_REGION).get_bucket_location(Bucket=S3_BUCKET).get("LocationConstraint")
    bucket_region = loc or "us-east-1"
except botocore.exceptions.ClientError:
    bucket_region = AWS_REGION

s3 = make_client(bucket_region)

# --- 接続確認（最低限） ---
try:
    s3.head_bucket(Bucket=S3_BUCKET)
except botocore.exceptions.ClientError as e:
    st.error("バケットにアクセスできません。バケット名 / リージョン / 権限をご確認ください。")
    st.code(f"{e.response['Error'].get('Code')} : {e.response['Error'].get('Message')}")
    st.stop()

def list_filenames_without_ext(prefix: str):
    """prefix 以下の .jsonl のベース名（拡張子除去）をアルファベット順で返す"""
    names = set()
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".jsonl") and not key.endswith("/"):
                    base = key.split("/")[-1].removesuffix(".jsonl")
                    names.add(base)
    except botocore.exceptions.ClientError as e:
        st.error(f"列挙失敗: {prefix} → {e.response['Error'].get('Code')}")
        st.code(e.response['Error'].get('Message'))
    return sorted(names)
    
def render_small_text(lines: list[str]):
    """小さくて薄い文字でリスト表示"""
    if not lines:
        st.markdown("<p style='color:gray; font-size:0.8em'>(なし)</p>", unsafe_allow_html=True)
    else:
        html = "<br>".join(lines)
        st.markdown(f"<p style='color:gray; font-size:0.8em'>{html}</p>", unsafe_allow_html=True)

# --- 取得 ---
mayor_list   = list_filenames_without_ext(PREFIX_MAYOR)
council_list = list_filenames_without_ext(PREFIX_COUNCIL)

# --- 表示（非表示バー） ---
st.markdown("本サイトでは以下の情報をもとに分析を行っています")

with st.expander("🏛️ きいてミライ｜市長発言AI分析", expanded=False):
    st.write(f"対象ファイル数: {len(mayor_list)}")
    render_small_text(mayor_list)

with st.expander("📜 きいてギカイ｜議会質疑AI分析", expanded=False):
    st.write(f"対象ファイル数: {len(council_list)}")
    render_small_text(council_list)

with st.expander("⚖️ ことばトレンド｜市政ワード分析", expanded=False):
    st.write("「ことばトレンド」ではきいてミライ、きいてギカイ双方の出典情報を集計対象にしています")

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
        ・ <a href="https://www.city.yamaguchi.yamaguchi.dbsr.jp/index.php/" target="_blank">
            山口市議会 議事録（公式HP）
          </a> <br>
        ・ <a href="https://www.city.yamaguchi.lg.jp/site/shicho/list68.html" target="_blank">
            山口市 市長の部屋 記者会見（市公式HP）
          </a> <br>
        ・ <a href="https://www.youtube.com/playlist?list=PLSBXr_PDKAbMOBbQdeQslWsrmSr-LyOdl" target="_blank">
            市長定例記者会見（市公式YouTube）
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



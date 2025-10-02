# -----------------------------
# imports
# -----------------------------
import streamlit as st
from pathlib import Path

APP_TREND_PATH   = "pages/01_ことばトレンド｜市政ワード分析.py"
APP_MAYOR_PATH   = "pages/02_きいてミライ｜市長発言AI分析.py"
APP_COUNCIL_PATH = "pages/03_きいてギカイ｜議会質疑AI分析.py"

# -----------------------------
# basic config
# -----------------------------
st.set_page_config(
    page_title="やまぐち市政ラボ｜市政をもっと身近に、市長と議会のことばアーカイブ＆リサーチ",
    page_icon="💬",
    layout="wide"
)

# -----------------------------
# simple router (switch_page)
# -----------------------------
slug_to_page = {
    "trend":   APP_TREND_PATH,
    "mayor":   APP_MAYOR_PATH,
    "council": APP_COUNCIL_PATH,
}

qp = st.query_params
goto = qp.get("goto")
if goto in slug_to_page:
    # クエリを消してから公式APIで内部遷移（戻る操作でも再実行が綺麗）
    st.query_params.clear()
    st.switch_page(slug_to_page[goto])

# -----------------------------
# styles (minimal)
# -----------------------------
st.markdown("""
<style>
:root{
  --blue-600:#2563EB;
  --ink-700:rgba(0,0,0,.75);
  --ink-650:rgba(0,0,0,.65);
  --ink-550:rgba(0,0,0,.55);
}

/* header */
.header{padding:14px 20px;border-radius:16px;
  background:linear-gradient(135deg,rgba(56,189,248,.16),rgba(59,130,246,.16));
  border:1px solid rgba(59,130,246,.22);box-shadow:0 6px 18px rgba(59,130,246,.10);}
.header .badge{display:inline-flex;gap:.4rem;align-items:center;
  background:#fff;border:1px solid rgba(0,0,0,.08);color:var(--ink-700);
  padding:.18rem .6rem;border-radius:999px;font-size:.85rem;font-weight:700;}
.header h1{font-size:1.5rem;line-height:1.2;margin:.2rem 0 .2rem;font-weight:800;}
.header .copy{color:var(--ink-700);font-size:1.02rem;line-height:1.6;margin:0;}

.small-muted{color:var(--ink-550);font-size:.92rem;margin:0 0; padding:1rem 1rem 1rem;}

/* card */
.card{padding:16px;border:1px solid rgba(0,0,0,.10);border-radius:14px;background:#fff;
  box-shadow:0 2px 6px rgba(0,0,0,.05);transition:transform .08s, box-shadow .12s, border-color .12s;}
.card:hover{transform:translateY(-2px);box-shadow:0 10px 22px rgba(0,0,0,.10);border-color:rgba(59,130,246,.35);}

/* kicker + title in the same block */
.card .top{margin:0;padding:0;}
.card .kicker{
  display:inline-block;padding:.14rem .5rem;border-radius:999px;
  border:1px solid #D5E2FF;background:#EEF3FF;color:#1a57ff;
  font-size:.86rem;font-weight:800;letter-spacing:.02em;margin:0;
}
.card .title{display:block;text-align:center;margin:.35rem 0 0;}
.card .title span{font-size:1.24rem;font-weight:800;color:#0b1b34;text-decoration:none;}

/* description */
.card .desc{color:var(--ink-650);line-height:1.6;font-size:.96rem;margin:.9rem 0 0;}

/* remove default p margins inside card */
.card p{margin:0;}
.card > div[data-testid="stMarkdownContainer"]{margin:0;padding:0;}

/* card-as-link wrapper */
a.cardlink{display:block; text-decoration:none; color:inherit;}
a.cardlink:hover .title span{ color:var(--blue-600); text-decoration:underline; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# card component (anchor-wrapped)
# -----------------------------
def card_with_link(slug: str, kicker: str, title: str, desc: str):
    href = f"?goto={slug}"
    # a.cardlink でカード全体をリンク化（ネストaを避けるため、タイトル内部は span）
    st.markdown(f"""
<a class="cardlink" href="{href}" target="_self">
  <div class="card">
    <div class="top">
      <span class="kicker">{kicker}</span>
      <span class="title"><span>{title}</span></span>
    </div>
    <p class="desc">{desc}</p>
  </div>
</a>
""", unsafe_allow_html=True)

# -----------------------------
# header
# -----------------------------
st.markdown("""
<div class="header">
  <span class="badge">💬 きいてポータル <span style="opacity:.6;">β</span></span>
  <h1>やまぐち市政ラボ</h1>
  <p class="copy">市政をもっと身近に。市長や議会のことばを、わかりやすく・身近に届けることを目指しています。</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="small-muted">本サイトは実験公開版です。今後の改善に向けて前向きなご意見をいただけますと幸いです。</p>', unsafe_allow_html=True)

# -----------------------------
# cards
# -----------------------------
c1, c2, c3 = st.columns(3, gap="large")
with c1:
    card_with_link(
        slug="trend",
        kicker="📊 市政ワード分析",
        title="ことばトレンド",
        desc="市政でよく使われる言葉を図表や時系列等で見える化。発言の傾向や注目テーマを知るきっかけに。"
    )
with c2:
    card_with_link(
        slug="mayor",
        kicker="👔 市長発言AI分析",
        title="きいてミライ",
        desc="市長の発言を生成AIが要約。市の方向性や重点施策を知ることで、まちの未来を考えるヒントに。"
    )
with c3:
    card_with_link(
        slug="council",
        kicker="🏛 議会質疑AI分析",
        title="きいてギカイ",
        desc="議会の質疑応答を生成AIが整理。議論の要点をまとめて、政策議論の全体像をつかむ手がかりに。"
    )

st.divider()

# -----------------------------
# about
# -----------------------------
st.subheader("このサイトについて")
st.markdown("""
本サイト（やまぐち市政ラボ）は、**市長や議員の発言を検索・分析できるアーカイブ＆リサーチサイト**です。  
まちの未来についての議論は生活のなかで触れる機会が少なく、街に愛着があっても内容は伝わりにくいものです。  
このサービスは、市政の公開情報をわかりやすく整理し、まちの今と未来をもっと身近に感じてもらえたらとの思いで作りました。  
""")

st.markdown("""
<div style="background:#f6f7f9;border:1px solid rgba(0,0,0,.08);color:rgba(0,0,0,.8);
padding:.9rem 1.1rem;border-radius:10px;margin:1rem 0;font-size:.92rem;line-height:1.6;">
議会や会見での発言は、活動のほんの一部に過ぎません。発言が少ないテーマであっても、実際には力を注いでいる方もいらっしゃいます。<br>
本サービスで公の情報に触れたあとは、ぜひ実際に会いに行くなど一次情報にも触れてみてください。<br>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# footer
# -----------------------------
st.markdown("---")
st.markdown("""
・ 本プロジェクトの詳細を知りたい方はこちら： [🔗 プロジェクト説明ページ（Notion）](https://fortune-orangutan-6aa.notion.site/1d0311267344808db873ff8af9b67365)  
・ 本プロジェクトは個人により運営されています。ご支援はこちら： [💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")
col_a, col_b, col_c = st.columns([1,1,1])
with col_a: st.write("© 2025 やまぐち市政ラボ")
with col_b: st.write("Made with Streamlit")

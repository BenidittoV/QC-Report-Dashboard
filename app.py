import json
import re
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="QC Report Dashboard",
    page_icon="https://oss-api.berijalan.id/web-berijalan/img/logo tab.webp",
    layout="wide",
)

# =========================
# CONSTANTS
# =========================
TL_COL = "metadata_teamLeader"
AGENT_COL = "metadata_namaAgent"
CALLRESULT_COL = "metadata_callResult"

DATE_COL = "call_date"              # format YYYY-MM-DD (opsional jika ada)
DATETIME_COL = "metadata_dateCall"  # contoh: "19 Februari 2026 08:07:05"
DURATION_COL = "duration"           # detik, contoh 40 / 120 / 95

SENTIMENT_CATEGORY_COL = "sentiment_category"
SENTIMENT_REASON_COL = "sentiment_reason"

# fallback jika sentiment_* tidak ada
LOV3_COL = "metadata_resultLov3"

DEFAULT_ALLOWED_CALL_TYPES = [
    "M1 (Setuju dikirim hitungan)",
    "M2 (Negosiasi)",
    "M3 (Setuju dengan hitungan)",
    "Tidak Minat",
    "Warm Leads",
    "Pencairan Minus",
]

ASPECT_COLUMNS_CANDIDATES = [
    "raw_data_greetings_open",
    "raw_data_say_acc",
    "raw_data_agent_name",
    "raw_data_cust_name",
    "raw_data_unit_cust",
    "raw_data_kontrak_cust",
    "raw_data_choice_cust",
    "raw_data_greetings_close",
    "raw_data_say_benefit",
    "raw_data_do_simulasi",
    "raw_data_say_include_angsuran",
    "raw_data_say_segmentation_offer_range",
    "raw_data_say_ref_contract_stat",
]

ASPECT_FRIENDLY_NAMES = {
    "raw_data_greetings_open": "Salam Pembuka",
    "raw_data_say_acc": "Menyebut ACC",
    "raw_data_agent_name": "Menyebut Nama Agent",
    "raw_data_cust_name": "Menyebut Nama Customer",
    "raw_data_unit_cust": "Menyebut Unit Customer",
    "raw_data_kontrak_cust": "Menyebut Kontrak Customer",
    "raw_data_choice_cust": "Menyebut Choice Customer",
    "raw_data_greetings_close": "Salam Penutup",
    "raw_data_say_benefit": "Menyebut Benefit",
    "raw_data_do_simulasi": "Melakukan Simulasi",
    "raw_data_say_include_angsuran": "Menyebut Termasuk Angsuran",
    "raw_data_say_segmentation_offer_range": "Segmentation Offer Range",
    "raw_data_say_ref_contract_stat": "Refer Contract Status",
}

LOADING_IMAGE_URL = "https://oss-api.berijalan.id/web-berijalan/img/logo tab.webp"

# =========================
# MODERN CSS + FULLSCREEN LOADER
# =========================
CUSTOM_CSS = f"""
<style>
html, body, [class*="css"]{{
  font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont;
}}
.stApp, .stApp * {{ color: #000000; }}
.stApp {{ background-color: #ffffff; }}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #002D72, #002D72);
  color: white !important;
  border-right: none;
}}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {{
  color: white !important;
}}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea {{
  background-color: rgba(255,255,255,0.14) !important;
  color: white !important;
  border-radius: 10px !important;
  border: none !important;
}}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
  color: rgba(255,255,255,0.88) !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{{
  background: rgba(255,255,255,0.14) !important;
  border: 1px solid rgba(255,255,255,0.40) !important;
  border-radius: 14px !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *{{ color: #ffffff !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button{{
  background: rgba(255,255,255,0.18) !important;
  border: 1px solid rgba(255,255,255,0.28) !important;
  color: #ffffff !important;
  border-radius: 12px !important;
}}

.hero {{
  background: #f8fafc;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 18px 20px;
}}
.hero-title {{
  font-size: 1.4rem;
  font-weight: 800;
  color: #000000;
  margin: 0;
}}
.hero-sub {{
  font-size: 0.95rem;
  color: #000000;
  opacity: 0.75;
  margin-top: 4px;
}}
.card {{
  background: #ffffff;
  border-radius: 16px;
  padding: 16px;
  border: 1px solid #e5e7eb;
  box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}}
.card h4 {{
  font-size: 0.9rem;
  margin-bottom: 6px;
  color: #000000;
  opacity: 0.70;
}}
.card .big {{
  font-size: 1.6rem;
  font-weight: 800;
  color: #000000;
  margin: 0;
}}
.badge {{
  display: inline-block;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 0.8rem;
  background: #eff6ff;
  color: #000000;
  border: 1px solid #bfdbfe;
  font-weight: 700;
}}
[data-testid="stDataFrame"] {{
  border-radius: 14px;
  border: 1px solid #e5e7eb;
  overflow: hidden;
}}
.stButton button {{
  background: #1d4ed8 !important;
  color: white !important;
  border-radius: 12px !important;
  padding: 0.6rem 1rem !important;
  font-weight: 700 !important;
  border: none !important;
}}
.stButton button:hover {{
  background: #1e40af !important;
  color: white !important;
}}
[data-testid="stMetric"] {{
  background: #ffffff;
  border-radius: 14px;
  padding: 12px;
  border: 1px solid #e5e7eb;
}}
[data-testid="stMetricValue"] {{
  font-weight: 900 !important;
  font-size: 28px !important;
  color: #000000 !important;
}}

button[data-baseweb="tab"] {{
  background: transparent;
  font-weight: 700;
  color: #000000;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: #000000;
  border-bottom: 3px solid #1d4ed8;
}}
details {{
  background: #f9fafb;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  padding: 8px;
}}
details * {{ color: #000000 !important; }}
.stCaption, .stCaption * {{ color: #000000 !important; opacity: 0.75; }}

.sidebar-logo {{
  position: fixed;
  bottom: 18px;
  left: 18px;
  width: 180px;
}}
.sidebar-logo p {{
  margin: 6px 0 0 0;
  font-size: 12px;
  color: rgba(255,255,255,0.88) !important;
}}
.vega-embed, .vega-embed * {{ background: #ffffff !important; }}
.vega-embed canvas, .vega-embed svg {{ background: #ffffff !important; }}

/* Radio -> Button Tabs */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"]{{
  display: flex !important;
  flex-direction: row !important;
  gap: 10px !important;
  width: 100% !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label{{
  flex: 1 1 0 !important;
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  background: rgba(255,255,255,0.14) !important;
  border: 1px solid rgba(255,255,255,0.35) !important;
  border-radius: 14px !important;
  padding: 10px 14px !important;
  margin: 0 !important;
  cursor: pointer !important;
  user-select: none !important;
  box-shadow: 0 6px 16px rgba(0,0,0,0.10) !important;
  transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease, border-color 120ms ease !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child{{
  display: none !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label *{{
  color: rgba(255,255,255,0.92) !important;
  font-weight: 800 !important;
  letter-spacing: 0.2px !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover{{
  background: rgba(255,255,255,0.22) !important;
  border-color: rgba(255,255,255,0.55) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 10px 22px rgba(0,0,0,0.16) !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:active{{
  transform: translateY(0px) scale(0.99) !important;
  box-shadow: 0 6px 14px rgba(0,0,0,0.12) !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked){{
  background: #ffffff !important;
  border-color: #ffffff !important;
  box-shadow: 0 12px 26px rgba(0,0,0,0.20) !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) *{{
  color: #002D72 !important;
  font-weight: 900 !important;
}}

/* LOADING OVERLAY */
#loading-overlay {{
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(255,255,255,0.92);
  z-index: 999999;
  display: none;
  align-items: center;
  justify-content: center;
}}

#loading-overlay .loading-inner {{
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
  padding: 22px 28px;
  border-radius: 20px;
  border: 1px solid #e5e7eb;
  background: rgba(255,255,255,0.92);
  box-shadow: 0 22px 54px rgba(0,0,0,0.12);
}}

#loading-overlay img.loading-logo {{
  width: 165px;
  height: 165px;
  border-radius: 999px;
  transform-origin: 50% 50%;
  will-change: transform;
  filter: drop-shadow(0 18px 36px rgba(0,0,0,0.16));
  animation: qcPulseSpin 1.70s linear infinite;
}}

@keyframes qcPulseSpin {{
  0%   {{ transform: scale(1.00) rotate(0deg); }}
  6%   {{ transform: scale(0.93) rotate(12deg); }}
  12%  {{ transform: scale(0.88) rotate(30deg); }}
  22%  {{ transform: scale(0.88) rotate(140deg); }}
  32%  {{ transform: scale(0.88) rotate(330deg); }}
  42%  {{ transform: scale(0.88) rotate(600deg); }}
  52%  {{ transform: scale(0.88) rotate(860deg); }}
  62%  {{ transform: scale(0.88) rotate(1000deg); }}
  70%  {{ transform: scale(0.88) rotate(1050deg); }}
  78%  {{ transform: scale(0.88) rotate(1070deg); }}
  86%  {{ transform: scale(1.08) rotate(1086deg); }}
  93%  {{ transform: scale(1.02) rotate(1092deg); }}
  100% {{ transform: scale(1.00) rotate(1080deg); }}
}}

@media (prefers-reduced-motion: reduce) {{
  #loading-overlay img.loading-logo {{
    animation: none !important;
    transform: scale(1) rotate(0);
  }}
}}

html:has(div[data-testid="stFileUploader"] div[role="progressbar"]) #loading-overlay,
body:has(div[data-testid="stFileUploader"] div[role="progressbar"]) #loading-overlay,
html:has(div[data-testid="stFileUploader"] progress) #loading-overlay,
body:has(div[data-testid="stFileUploader"] progress) #loading-overlay,
html:has(div[data-testid="stFileUploader"] [data-testid="stSpinner"]) #loading-overlay,
body:has(div[data-testid="stFileUploader"] [data-testid="stSpinner"]) #loading-overlay,
html:has(div[data-testid="stFileUploader"] [data-testid="stFileUploaderProgress"]) #loading-overlay,
body:has(div[data-testid="stFileUploader"] [data-testid="stFileUploaderProgress"]) #loading-overlay,
html:has(div[data-testid="stFileUploader"] [data-testid="stProgress"]) #loading-overlay,
body:has(div[data-testid="stFileUploader"] [data-testid="stProgress"]) #loading-overlay {{
  display: flex;
}}

html:has(#processing-flag) #loading-overlay,
body:has(#processing-flag) #loading-overlay {{
  display: flex;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    f"""
    <div id="loading-overlay">
      <div class="loading-inner">
        <img class="loading-logo" src="{LOADING_IMAGE_URL}" />
        <div class="loading-text">Memproses file…</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            def _coerce(x):
                if pd.isna(x):
                    return ""
                if isinstance(x, (bytes, bytearray)):
                    return x.decode("utf-8", errors="ignore")
                if isinstance(x, (dict, list, tuple, set)):
                    return json.dumps(list(x) if isinstance(x, set) else x, ensure_ascii=False)
                return str(x)
            out[col] = out[col].map(_coerce)
    return out

def normalize_to_binary(v) -> int:
    if pd.isna(v):
        return 0
    if isinstance(v, (int, float, np.integer, np.floating)):
        return 1 if float(v) > 0 else 0
    s = str(v).strip().lower()
    if s in {"", "nan", "none", "0"}:
        return 0
    if s == "1":
        return 1
    return 1

def tier_from_percent(p: float) -> str:
    if p >= 90:
        return "S"
    if p >= 80:
        return "A"
    if p >= 70:
        return "B"
    if p >= 60:
        return "C"
    return "D"

def safe_pct(binary_series: pd.Series) -> float:
    if binary_series is None or len(binary_series) == 0:
        return np.nan
    return round(float(binary_series.mean() * 100.0), 2)

def grade_badge(grade: str) -> str:
    if grade == "S":
        return "🏆 S"
    if grade == "A":
        return "✅ A"
    if grade == "B":
        return "👍 B"
    if grade == "C":
        return "⚠️ C"
    if grade == "D":
        return "⛔ D"
    return grade

@st.cache_data
def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Format file tidak didukung. Upload CSV atau XLSX.")

def week_ranges_sun_sat_for_month(year: int, month: int):
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end = month_start + pd.offsets.MonthEnd(1)
    days_to_sun = (6 - month_start.weekday()) % 7
    first_sunday = month_start + pd.Timedelta(days=days_to_sun)

    ranges = []
    if first_sunday > month_start:
        ranges.append((month_start, first_sunday - pd.Timedelta(days=1)))

    cur = first_sunday
    while cur <= month_end:
        start = cur
        end = min(cur + pd.Timedelta(days=6), month_end)
        if start == end and start.weekday() == 6:
            break
        ranges.append((start, end))
        cur = cur + pd.Timedelta(days=7)
    return ranges

def light_table(df: pd.DataFrame):
    df = make_arrow_safe(df)
    fmt = {}
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            fmt[c] = "{:,.0f}"
        elif pd.api.types.is_float_dtype(df[c]):
            fmt[c] = "{:.2f}"

    return (
        df.style
        .format(fmt, na_rep="")
        .set_properties(**{
            "background-color": "white",
            "color": "black",
            "border-color": "#e5e7eb",
            "font-size": "13px",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#0A3D91"),
                ("color", "white"),
                ("border", "1px solid #e5e7eb"),
                ("font-weight", "700"),
                ("text-align", "left"),
                ("padding", "10px"),
            ]},
            {"selector": "td", "props": [
                ("border", "1px solid #e5e7eb"),
                ("padding", "8px"),
            ]},
            {"selector": "table", "props": [
                ("border-collapse", "collapse"),
                ("width", "100%"),
            ]},
        ])
    )

def normalize_identity_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TL_COL] = out[TL_COL].astype(str).str.strip()
    out[AGENT_COL] = out[AGENT_COL].astype(str).str.strip()
    out = out[(out[TL_COL] != "") & (out[TL_COL].str.lower() != "nan")]
    out = out[(out[AGENT_COL] != "") & (out[AGENT_COL].str.lower() != "nan")]
    return out

def filter_call_types(df: pd.DataFrame) -> pd.DataFrame:
    if CALLRESULT_COL not in df.columns:
        st.warning(f"Kolom `{CALLRESULT_COL}` tidak ditemukan. Scoring dilakukan tanpa filter call type.")
        return df
    allowed_call_lower = {str(x).strip().lower() for x in DEFAULT_ALLOWED_CALL_TYPES}
    call_series = df[CALLRESULT_COL].astype(str).str.strip()
    return df[call_series.str.lower().isin(allowed_call_lower)].copy()

def parse_metadata_datecall(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    bulan = {
        "januari": "01", "februari": "02", "maret": "03", "april": "04",
        "mei": "05", "juni": "06", "juli": "07", "agustus": "08",
        "september": "09", "oktober": "10", "november": "11", "desember": "12",
    }
    s_low = s.str.lower()
    for nama, mm in bulan.items():
        s_low = s_low.str.replace(fr"\b{nama}\b", mm, regex=True)
    s_low = s_low.str.replace(r"^(\d{1,2})\s+(\d{2})\s+(\d{4})\s+", r"\1-\2-\3 ", regex=True)
    return pd.to_datetime(s_low, format="%d-%m-%Y %H:%M:%S", errors="coerce")

def ensure_date_and_dt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if DATETIME_COL in out.columns:
        out["_dt_call"] = parse_metadata_datecall(out[DATETIME_COL])
    else:
        out["_dt_call"] = pd.NaT

    if DATE_COL in out.columns:
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    else:
        if out["_dt_call"].notna().any():
            out[DATE_COL] = out["_dt_call"].dt.floor("D")
        else:
            st.error(f"Tidak ada `{DATE_COL}` dan `{DATETIME_COL}` juga tidak tersedia/valid. Minimal salah satu harus ada.")
            st.stop()

    out = out.dropna(subset=[DATE_COL])
    if out.empty:
        st.warning("Semua tanggal gagal diparse. Pastikan format tanggal valid.")
        st.stop()

    return out

def compute_overall_from_aspects(df_subset: pd.DataFrame, aspect_cols: list[str]) -> float:
    if df_subset.empty:
        return np.nan
    vals = [safe_pct(df_subset[c].apply(normalize_to_binary)) for c in aspect_cols]
    return round(float(np.nanmean(vals)), 2) if len(vals) else np.nan

def count_and_total(df_subset: pd.DataFrame, col: str) -> tuple[int, int]:
    total = int(len(df_subset))
    if total == 0:
        return 0, 0
    hit = int(df_subset[col].apply(normalize_to_binary).sum())
    return hit, total

def format_hit_total(hit: int, total: int) -> str:
    if total <= 0:
        return "0/0"
    return f"{hit}/{total}"

# ===== interest helpers =====
def _norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def _compact(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum())

def is_warmleads_value(x) -> bool:
    v = _compact(_norm_text(x))
    return v in {"warmleads", "warmlead"}

def agent_positive_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower()
    m_mask = s.str.contains(r"\bm1\b|\bm2\b|\bm3\b", na=False)
    w_mask = s.str.contains(r"warm\s*leads?|warmleads?", na=False)
    return m_mask | w_mask

def interest_masks(df: pd.DataFrame):
    req_sent = [SENTIMENT_CATEGORY_COL, SENTIMENT_REASON_COL, CALLRESULT_COL]
    if all(c in df.columns for c in req_sent):
        dfx = df.copy()
        dfx[SENTIMENT_CATEGORY_COL] = dfx[SENTIMENT_CATEGORY_COL].astype(str).str.strip().str.lower()
        dfx[SENTIMENT_REASON_COL] = dfx[SENTIMENT_REASON_COL].astype(str).str.lower()

        ai_mask = dfx[SENTIMENT_CATEGORY_COL].isin(["potensi berminat", "ragu-ragu"])
        agent_mask = agent_positive_mask(dfx[CALLRESULT_COL])

        keywords = ["pikir-pikir", "pikir pikir", "pikir", "pertimbangkan", "diskusi", "diskusikan"]
        pattern = "|".join(keywords)
        rule3 = (
            (~agent_mask)
            & (dfx[SENTIMENT_CATEGORY_COL] == "tidak berminat")
            & (dfx[SENTIMENT_REASON_COL].str.contains(pattern, na=False))
        )

        actual_mask = ai_mask | agent_mask | rule3
        return {"mode": "sentiment", "ai_mask": ai_mask, "agent_mask": agent_mask, "actual_mask": actual_mask}

    req_lov3 = [LOV3_COL, CALLRESULT_COL]
    if all(c in df.columns for c in req_lov3):
        dfx = df.copy()
        ai_mask = dfx[LOV3_COL].apply(is_warmleads_value)
        agent_mask = agent_positive_mask(dfx[CALLRESULT_COL])
        actual_mask = ai_mask | agent_mask
        return {"mode": "lov3", "ai_mask": ai_mask, "agent_mask": agent_mask, "actual_mask": actual_mask}

    return None

def build_agent_interest_summary(scope_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if AGENT_COL not in scope_df.columns:
        return None
    im = interest_masks(scope_df)
    if im is None:
        return None
    actual_mask = im["actual_mask"]

    d = scope_df.copy()
    d[AGENT_COL] = d[AGENT_COL].astype(str).str.strip()
    d = d[(d[AGENT_COL] != "") & (d[AGENT_COL].str.lower() != "nan")]

    m = actual_mask.reindex(d.index).fillna(False).astype(bool)
    d["_is_interest"] = m.values

    summary = (
        d.groupby(AGENT_COL, as_index=False)
        .agg(
            jumlah_minat=("_is_interest", "sum"),
            jumlah_rekaman=("_is_interest", "size"),
        )
    )
    summary["jumlah_tidak_minat"] = summary["jumlah_rekaman"] - summary["jumlah_minat"]

    summary = summary.rename(columns={
        AGENT_COL: "Agent",
        "jumlah_minat": "Jumlah Minat",
        "jumlah_tidak_minat": "Jumlah Tidak Minat",
        "jumlah_rekaman": "Jumlah Rekaman",
    }).sort_values(by=["Jumlah Minat", "Jumlah Rekaman"], ascending=[False, False])

    for c in ["Jumlah Minat", "Jumlah Tidak Minat", "Jumlah Rekaman"]:
        summary[c] = pd.to_numeric(summary[c], errors="coerce").fillna(0).astype(int)

    return summary

def build_agent_daily_presence_summary(
    scope_df: pd.DataFrame,
    actual_mask: pd.Series,
    active_days: list,
    allowed_missing_days: int = 2,
    invert: bool = False,
) -> Optional[pd.DataFrame]:
    if AGENT_COL not in scope_df.columns or scope_df.empty:
        return None

    d = scope_df.copy()
    d[AGENT_COL] = d[AGENT_COL].astype(str).str.strip()
    d = d[(d[AGENT_COL] != "") & (d[AGENT_COL].str.lower() != "nan")].copy()
    if d.empty:
        return None

    d["_day_dt"] = pd.to_datetime(d[DATE_COL]).dt.floor("D")
    d["_is_interest"] = actual_mask.loc[d.index].values

    total_active_days = len(active_days)
    if total_active_days == 0:
        return None

    rows = []
    for agent, da in d.groupby(AGENT_COL):
        hadir_days = int(da["_day_dt"].nunique())
        kosong_days = total_active_days - hadir_days
        jumlah_rekaman = int(len(da))
        jumlah_minat = int(da["_is_interest"].sum())
        jumlah_tidak_minat = jumlah_rekaman - jumlah_minat

        if invert:
            keep = kosong_days > allowed_missing_days
        else:
            keep = kosong_days <= allowed_missing_days

        if keep:
            rows.append({
                "Agent": agent,
                "Hari Hadir": hadir_days,
                "Hari Kosong": kosong_days,
                "Jumlah Minat": jumlah_minat,
                "Jumlah Tidak Minat": jumlah_tidak_minat,
                "Jumlah Rekaman": jumlah_rekaman,
            })

    if not rows:
        return None

    out = pd.DataFrame(rows).sort_values(
        by=["Jumlah Minat", "Jumlah Rekaman", "Hari Kosong"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return out

def build_daily_interest_chart_for_agent_group(
    scope_df: pd.DataFrame,
    actual_mask: pd.Series,
    selected_agents: set[str],
) -> Optional[pd.DataFrame]:
    if not selected_agents:
        return None

    d = scope_df.copy()
    d[AGENT_COL] = d[AGENT_COL].astype(str).str.strip()
    d = d[d[AGENT_COL].isin(selected_agents)].copy()
    if d.empty:
        return None

    d["_day_dt"] = pd.to_datetime(d[DATE_COL]).dt.floor("D")
    d["_is_interest"] = actual_mask.loc[d.index].values

    daily = (
        d.groupby("_day_dt", as_index=False)
        .agg(
            jumlah_minat=("_is_interest", "sum"),
            jumlah_rekaman=("_is_interest", "size"),
        )
    )
    daily["rate_minat"] = np.where(
        daily["jumlah_rekaman"] > 0,
        (daily["jumlah_minat"] / daily["jumlah_rekaman"] * 100.0),
        np.nan,
    ).round(2)

    return daily

def split_calendar_month_ranges(year: int, month: int) -> dict:
    last_day = int(pd.Timestamp(year=year, month=month, day=1).days_in_month)

    if last_day == 31:
        p1_start, p1_end = 1, 16
        p2_start, p2_end = 17, 31
    else:
        p1_start, p1_end = 1, 15
        p2_start, p2_end = 16, last_day

    return {
        "last_day": last_day,
        "p1_start": p1_start,
        "p1_end": p1_end,
        "p2_start": p2_start,
        "p2_end": p2_end,
        "label_1": f"Periode 1 ({p1_start}–{p1_end})",
        "label_2": f"Periode 2 ({p2_start}–{p2_end})",
    }

def build_daily_interest_for_period(
    scope_df: pd.DataFrame,
    actual_mask: pd.Series,
    start_day: int,
    end_day: int,
) -> Optional[pd.DataFrame]:
    if actual_mask is None or scope_df.empty:
        return None

    d = scope_df.copy()
    d["_day_dt"] = pd.to_datetime(d[DATE_COL]).dt.floor("D")
    d["_day_num"] = d["_day_dt"].dt.day
    d = d[(d["_day_num"] >= start_day) & (d["_day_num"] <= end_day)].copy()

    if d.empty:
        return None

    d["_is_interest"] = actual_mask.loc[d.index].values

    daily = (
        d.groupby("_day_dt", as_index=False)
        .agg(
            jumlah_minat=("_is_interest", "sum"),
            jumlah_rekaman=("_is_interest", "size"),
        )
    )

    daily["rate_minat"] = np.where(
        daily["jumlah_rekaman"] > 0,
        (daily["jumlah_minat"] / daily["jumlah_rekaman"] * 100.0),
        np.nan,
    ).round(2)

    return daily.sort_values("_day_dt").reset_index(drop=True)

def _call_result_norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()

def build_priority_followup_tables(scope_df: pd.DataFrame) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if CALLRESULT_COL not in scope_df.columns:
        return None, None

    dfx = scope_df.copy()
    dfx[CALLRESULT_COL] = dfx[CALLRESULT_COL].astype(str).str.strip()

    aspek_tabel_1 = [
        "raw_data_choice_cust",
        "raw_data_say_include_angsuran",
        "raw_data_say_segmentation_offer_range",
        "raw_data_say_benefit",
        "raw_data_do_simulasi",
    ]
    aspek_tabel_2 = [
        "raw_data_choice_cust",
        "raw_data_say_include_angsuran",
        "raw_data_do_simulasi",
    ]

    aspek_tabel_1 = [c for c in aspek_tabel_1 if c in dfx.columns]
    aspek_tabel_2 = [c for c in aspek_tabel_2 if c in dfx.columns]

    call_norm = _call_result_norm(dfx[CALLRESULT_COL])

    customer_id_candidates = [
        "metadata_idCustomer",
        "metadata_customerId",
        "customer_id",
        "cust_id",
        "id_customer",
        "raw_data_kontrak_cust",
    ]
    customer_id_col = next((c for c in customer_id_candidates if c in dfx.columns), None)

    t1 = None
    mask_m123 = call_norm.str.contains(r"\bm1\b|\bm2\b|\bm3\b", na=False)

    if aspek_tabel_1:
        d1 = dfx[mask_m123].copy()

        if not d1.empty:
            def _missing_aspects_row(row):
                missing = []
                for col in aspek_tabel_1:
                    if normalize_to_binary(row[col]) == 0:
                        missing.append(ASPECT_FRIENDLY_NAMES.get(col, col))
                return ", ".join(missing)

            d1["Aspek Jarang Disebut"] = d1.apply(_missing_aspects_row, axis=1)
            d1 = d1[d1["Aspek Jarang Disebut"].str.strip() != ""].copy()

            rename_map = {
                AGENT_COL: "Nama Agent",
                DATETIME_COL: "Tanggal Call",
            }
            if customer_id_col:
                rename_map[customer_id_col] = "ID Customer"

            selected_cols = [AGENT_COL]
            if customer_id_col:
                selected_cols.append(customer_id_col)
            if DATETIME_COL in d1.columns:
                selected_cols.append(DATETIME_COL)
            selected_cols.append("Aspek Jarang Disebut")

            t1 = d1[selected_cols].rename(columns=rename_map).copy()

    t2 = None
    mask_tidak_minat = call_norm.str.contains(r"tidak\s*minat", na=False)

    if aspek_tabel_2:
        d2 = dfx[mask_tidak_minat].copy()

        if not d2.empty:
            def _present_aspects_row(row):
                present = []
                for col in aspek_tabel_2:
                    if normalize_to_binary(row[col]) == 1:
                        present.append(ASPECT_FRIENDLY_NAMES.get(col, col))
                return ", ".join(present)

            d2["Aspek Sudah Disebut"] = d2.apply(_present_aspects_row, axis=1)
            d2 = d2[d2["Aspek Sudah Disebut"].str.strip() != ""].copy()

            rename_map = {
                AGENT_COL: "Nama Agent",
                DATETIME_COL: "Tanggal Call",
            }
            if customer_id_col:
                rename_map[customer_id_col] = "ID Customer"

            selected_cols = [AGENT_COL]
            if customer_id_col:
                selected_cols.append(customer_id_col)
            if DATETIME_COL in d2.columns:
                selected_cols.append(DATETIME_COL)
            selected_cols.append("Aspek Sudah Disebut")

            t2 = d2[selected_cols].rename(columns=rename_map).copy()

    return t1, t2

def build_tl_agent_comparison_tables(
    scope_df: pd.DataFrame,
    aspect_cols: list[str],
    top_n_aspects: int = 4,
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[dict]
]:
    if AGENT_COL not in scope_df.columns or scope_df.empty:
        return None, None, None, None, None

    d = scope_df.copy()
    d[AGENT_COL] = d[AGENT_COL].astype(str).str.strip()
    d = d[(d[AGENT_COL] != "") & (d[AGENT_COL].str.lower() != "nan")].copy()

    if d.empty:
        return None, None, None, None, None

    rows = []
    for agent, da in d.groupby(AGENT_COL):
        agent_overall = compute_overall_from_aspects(da, aspect_cols)
        if np.isnan(agent_overall):
            continue

        aspek_scores = []
        for col in aspect_cols:
            bin_series = da[col].apply(normalize_to_binary)
            hit = int(bin_series.sum())
            total = int(len(da))
            pct = safe_pct(bin_series)

            if not np.isnan(pct):
                aspek_scores.append({
                    "col": col,
                    "aspek": ASPECT_FRIENDLY_NAMES.get(col, col),
                    "pct": float(pct),
                    "hit": hit,
                    "total": total,
                })

        if not aspek_scores:
            continue

        aspek_scores_sorted_asc = sorted(aspek_scores, key=lambda x: (x["pct"], x["aspek"]))
        aspek_scores_sorted_desc = sorted(aspek_scores, key=lambda x: (-x["pct"], x["aspek"]))

        weak_n = ", ".join(
            [f'{x["aspek"]} ({x["hit"]}/{x["total"]})' for x in aspek_scores_sorted_asc[:top_n_aspects]]
        )
        strong_n = ", ".join(
            [f'{x["aspek"]} ({x["hit"]}/{x["total"]})' for x in aspek_scores_sorted_desc[:top_n_aspects]]
        )

        rows.append({
            "Mitra (Agent)": agent,
            "Jumlah Rekaman": int(len(da)),
            "Overall Agent (%)": round(float(agent_overall), 2),
            "4 Aspek yang Jarang Disebutkan": weak_n,
            "4 Aspek yang Paling Konsisten Disebutkan": strong_n,
            "_scores_asc": aspek_scores_sorted_asc,
            "_scores_desc": aspek_scores_sorted_desc,
        })

    if not rows:
        return None, None, None, None, None

    summary = pd.DataFrame(rows)

    summary_worst_sorted = summary.sort_values(
        by=["Overall Agent (%)", "Jumlah Rekaman"],
        ascending=[True, False],
    ).reset_index(drop=True)

    summary_best_sorted = summary.sort_values(
        by=["Overall Agent (%)", "Jumlah Rekaman"],
        ascending=[False, False],
    ).reset_index(drop=True)

    worst_table = (
        summary_worst_sorted[["Mitra (Agent)", "4 Aspek yang Jarang Disebutkan"]]
        .rename(columns={"Mitra (Agent)": "Nama Mitra (Agent)"})
        .copy()
    )

    best_table = (
        summary_best_sorted[["Mitra (Agent)", "4 Aspek yang Paling Konsisten Disebutkan"]]
        .rename(columns={"Mitra (Agent)": "Nama Mitra (Agent)"})
        .copy()
    )

    best_agent_row = summary_best_sorted.iloc[0]
    worst_agent_row = summary_worst_sorted.iloc[0]

    best_name = best_agent_row["Mitra (Agent)"]
    worst_name = worst_agent_row["Mitra (Agent)"]

    best_overall = float(best_agent_row["Overall Agent (%)"])
    worst_overall = float(worst_agent_row["Overall Agent (%)"])
    gap_overall = round(best_overall - worst_overall, 2)

    score_best_map = {}
    for item in best_agent_row["_scores_desc"]:
        score_best_map[item["col"]] = item

    score_worst_map = {}
    for item in worst_agent_row["_scores_asc"]:
        score_worst_map[item["col"]] = item

    compare_rows = []
    for col in aspect_cols:
        best_item = score_best_map.get(col)
        worst_item = score_worst_map.get(col)
        if best_item is None or worst_item is None:
            continue

        gap_aspek = round(float(best_item["pct"] - worst_item["pct"]), 2)
        compare_rows.append({
            "Aspek": ASPECT_FRIENDLY_NAMES.get(col, col),
            f"{best_name}": f'{best_item["hit"]}/{best_item["total"]} ({best_item["pct"]:.2f}%)',
            f"{worst_name}": f'{worst_item["hit"]}/{worst_item["total"]} ({worst_item["pct"]:.2f}%)',
            "Gap (%)": gap_aspek,
        })

    comparison_detail_df = pd.DataFrame(compare_rows).sort_values(
        by="Gap (%)", ascending=False
    ).reset_index(drop=True)

    comparison_summary_df = pd.DataFrame([{
        "Mitra Terbaik": best_name,
        "Overall Terbaik (%)": round(best_overall, 2),
        "Mitra Terburuk": worst_name,
        "Overall Terburuk (%)": round(worst_overall, 2),
        "Gap Overall (%)": gap_overall,
    }])

    insight = None
    if not comparison_detail_df.empty:
        top_gap_row = comparison_detail_df.iloc[0].to_dict()
        insight = {
            "best_name": best_name,
            "worst_name": worst_name,
            "gap_overall": gap_overall,
            "top_gap_aspect": top_gap_row.get("Aspek"),
            "top_gap_value": top_gap_row.get("Gap (%)"),
        }

    return worst_table, best_table, comparison_summary_df, comparison_detail_df, insight

def normalize_call_bucket(v) -> Optional[str]:
    s = str(v).strip().lower()
    if re.search(r"\bm1\b", s):
        return "M1"
    if re.search(r"\bm2\b", s):
        return "M2"
    if re.search(r"\bm3\b", s):
        return "M3"
    if re.search(r"tidak\s*minat", s):
        return "Tidak Minat"
    return None

def parse_duration_seconds(v) -> float:
    if pd.isna(v):
        return np.nan

    if isinstance(v, (int, float, np.integer, np.floating)):
        val = float(v)
        return val if val >= 0 else np.nan

    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan

    if ":" in s:
        parts = s.split(":")
        try:
            parts = [int(float(x)) for x in parts]
            if len(parts) == 2:
                return float(parts[0] * 60 + parts[1])
            if len(parts) == 3:
                return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
        except Exception:
            return np.nan

    try:
        val = float(s.replace(",", "."))
        return val if val >= 0 else np.nan
    except Exception:
        return np.nan

def format_seconds_mmss(v) -> str:
    if pd.isna(v):
        return "-"
    total = int(round(float(v)))
    menit = total // 60
    detik = total % 60
    return f"{menit}:{detik:02d}"

# ===== chart helpers =====
def style_chart(chart, height: int):
    return (
        chart.properties(height=height, background="white")
        .configure_view(stroke=None, fill="white")
        .configure_axis(labelColor="black", titleColor="black", gridColor="#e5e7eb")
    )

def _infer_entity_type(key_prefix: str) -> str:
    if key_prefix.lower().startswith("agent"):
        return "Agent"
    return "TL"

def _get_entity_col(entity_type: str) -> str:
    return AGENT_COL if entity_type == "Agent" else TL_COL

def _highlight_rows_by_aspek(styler: Styler, aspek_set: set[str]):
    def _row_style(row):
        aspek = str(row.get("Aspek", ""))
        if aspek in aspek_set:
            return ["background-color: #FCA5A5; color: #7F1D1D; font-weight: 700;"] * len(row)
        return [""] * len(row)
    return styler.apply(_row_style, axis=1)

def compute_hourly_reference_lines(
    baseline_df: pd.DataFrame,
    aspect_cols: list[str],
    entity_col: str,
) -> dict:
    result = {"upper": np.nan, "lower": np.nan, "kkm": np.nan}

    if baseline_df is None or baseline_df.empty or entity_col not in baseline_df.columns:
        return result

    dfx = baseline_df.copy()
    dfx = ensure_date_and_dt(dfx)

    if "_dt_call" not in dfx.columns or dfx["_dt_call"].notna().sum() == 0:
        return result

    dfx[entity_col] = dfx[entity_col].astype(str).str.strip()
    dfx = dfx[(dfx[entity_col] != "") & (dfx[entity_col].str.lower() != "nan")].copy()
    if dfx.empty:
        return result

    dfx["_hour"] = dfx["_dt_call"].dt.hour
    dfx = dfx[(dfx["_hour"] >= 8) & (dfx["_hour"] <= 17) & (dfx["_hour"] != 12)].copy()
    if dfx.empty:
        return result

    entity_hour_values = []
    for (_, _), dsub in dfx.groupby([entity_col, "_hour"]):
        ov = compute_overall_from_aspects(dsub, aspect_cols)
        if not np.isnan(ov):
            entity_hour_values.append(float(ov))

    if not entity_hour_values:
        return result

    result["upper"] = float(np.nanmax(entity_hour_values))
    result["lower"] = float(np.nanmin(entity_hour_values))
    result["kkm"] = float(np.nanmean(entity_hour_values))
    return result

# =========================
# CORE RENDER BLOCK
# =========================
def run_performance_block(
    df_base: pd.DataFrame,
    header_badges_html: str,
    title_context: str,
    key_prefix: str,
    show_agent_interest_table_in_tab3: bool = False,
    baseline_df: Optional[pd.DataFrame] = None,
):
    df_base = filter_call_types(df_base)
    if df_base.empty:
        st.warning("Tidak ada data setelah filter call type.")
        st.stop()

    df_base = ensure_date_and_dt(df_base)
    baseline_df = df_base if baseline_df is None else ensure_date_and_dt(filter_call_types(baseline_df))

    unique_days = int(df_base[DATE_COL].dt.date.nunique())
    time_mode = "Harian" if unique_days <= 1 else "Bulanan"

    aspect_cols = [c for c in ASPECT_COLUMNS_CANDIDATES if c in df_base.columns]
    missing_aspects = [c for c in ASPECT_COLUMNS_CANDIDATES if c not in df_base.columns]
    if not aspect_cols:
        st.error("Tidak ada kolom aspek yang ditemukan di file untuk dihitung.")
        st.stop()

    entity_type = _infer_entity_type(key_prefix)
    entity_col = _get_entity_col(entity_type)

    selected_period_label = ""
    selected_month = None

    # =========================
    # BUILD RESULT TABLE
    # =========================
    if time_mode == "Bulanan":
        df_base["_month"] = df_base[DATE_COL].dt.to_period("M").astype(str)
        month_list = sorted(df_base["_month"].unique().tolist())

        selected_month = st.selectbox(
            "Bulan & Tahun (terdeteksi)",
            month_list,
            index=len(month_list) - 1,
            key=f"{key_prefix}_month",
        )
        selected_period_label = selected_month

        dfm = df_base[df_base["_month"] == selected_month].copy()
        if dfm.empty:
            st.warning("Tidak ada data untuk bulan terpilih.")
            st.stop()

        yy, mm = selected_month.split("-")
        yy, mm = int(yy), int(mm)
        week_ranges = week_ranges_sun_sat_for_month(yy, mm)

        rows = []
        for col in aspect_cols:
            row = {"Aspek": ASPECT_FRIENDLY_NAMES.get(col, col)}
            for i, (ws, we) in enumerate(week_ranges, start=1):
                df_w = dfm[(dfm[DATE_COL] >= ws) & (dfm[DATE_COL] <= we)]
                hit, total = count_and_total(df_w, col)
                row[f"Minggu {i}"] = format_hit_total(hit, total)

            hit_m, total_m = count_and_total(dfm, col)
            pct_m = safe_pct(dfm[col].apply(normalize_to_binary))
            row["Bulanan"] = format_hit_total(hit_m, total_m)
            row["_pct_main"] = pct_m if not np.isnan(pct_m) else np.nan
            row["Grade"] = tier_from_percent(pct_m) if not np.isnan(pct_m) else "-"
            rows.append(row)

        result_df = pd.DataFrame(rows).sort_values(by="_pct_main", ascending=True)
        overall_value = round(float(pd.to_numeric(result_df["_pct_main"], errors="coerce").mean()), 2)
        scope_df = dfm

    else:
        if df_base["_dt_call"].notna().sum() == 0:
            st.error(f"Mode Harian membutuhkan `{DATETIME_COL}` yang valid (contoh: 19 Februari 2026 08:07:05).")
            st.stop()

        only_day = df_base[DATE_COL].dt.date.unique()
        only_day = only_day[0] if len(only_day) else "-"
        selected_period_label = str(only_day)

        dfm = df_base.copy()
        buckets = [
            ("08–10", 8, 10),
            ("10–12", 10, 12),
            ("13–15", 13, 15),
            ("15–17", 15, 17),
        ]

        dfm["_hour"] = dfm["_dt_call"].dt.hour
        outside_mask = (dfm["_hour"] < 8) | (dfm["_hour"] >= 17)
        has_outside = bool(outside_mask.any())

        rows = []
        for col in aspect_cols:
            row = {"Aspek": ASPECT_FRIENDLY_NAMES.get(col, col)}
            for label, h0, h1 in buckets:
                d = dfm[(dfm["_hour"] >= h0) & (dfm["_hour"] < h1)]
                hit, total = count_and_total(d, col)
                row[label] = format_hit_total(hit, total)

            if has_outside:
                d_out = dfm[outside_mask]
                hit_out, total_out = count_and_total(d_out, col)
                row["Di luar 08–17"] = format_hit_total(hit_out, total_out)

            hit_d, total_d = count_and_total(dfm, col)
            pct_d = safe_pct(dfm[col].apply(normalize_to_binary))
            row["Harian"] = format_hit_total(hit_d, total_d)
            row["_pct_main"] = pct_d if not np.isnan(pct_d) else np.nan
            row["Grade"] = tier_from_percent(pct_d) if not np.isnan(pct_d) else "-"
            rows.append(row)

        result_df = pd.DataFrame(rows).sort_values(by="_pct_main", ascending=True)
        overall_value = round(float(pd.to_numeric(result_df["_pct_main"], errors="coerce").mean()), 2)
        scope_df = dfm

    # =========================
    # KPI HEADER
    # =========================
    period_badges = f"""
&nbsp; <span class="badge">Mode: <b>{time_mode}</b></span>
&nbsp; <span class="badge">Periode: <b>{selected_period_label}</b></span>
"""
    left, right = st.columns([1.4, 1.0], vertical_alignment="center")
    with left:
        st.markdown(
            f"""<div class="card">
<h4>{title_context}</h4>
{header_badges_html}
{period_badges}
</div>""",
            unsafe_allow_html=True,
        )
    with right:
        c1, c2, c3 = st.columns(3)
        label_overall = "Overall Bulanan" if time_mode == "Bulanan" else "Overall Harian"
        c1.markdown(f'<div class="card"><h4>{label_overall}</h4><p class="big">{overall_value:.2f}%</p></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="card"><h4>Jumlah Rekaman</h4><p class="big">{len(scope_df)}</p></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="card"><h4>Aspek Dihitung</h4><p class="big">{len(aspect_cols)}</p></div>', unsafe_allow_html=True)

    st.write("")
    if missing_aspects:
        with st.expander("ℹ️ Beberapa kolom aspek tidak ditemukan (aman, hanya di-skip)"):
            st.write(missing_aspects)

    # =========================
    # KPI MINAT
    # =========================
    im = interest_masks(scope_df)
    ai_mask = agent_mask = actual_mask = None
    interest_mode = None

    if im is None:
        st.info(
            "Kolom untuk hitung minat tidak ditemukan. "
            f"Butuh: `{SENTIMENT_CATEGORY_COL}`+`{SENTIMENT_REASON_COL}` atau `{LOV3_COL}`, dan `{CALLRESULT_COL}`."
        )
    else:
        interest_mode = im["mode"]
        ai_mask = im["ai_mask"]
        agent_mask = im["agent_mask"]
        actual_mask = im["actual_mask"]

        if interest_mode == "lov3":
            warm_cnt = int(ai_mask.sum())
            not_cnt = int((~ai_mask).sum())
            m1, m2 = st.columns(2)
            m1.metric("Warm Leads / Minat (AI Lov3)", warm_cnt)
            m2.metric("Tidak Minat (AI Lov3)", not_cnt)
        else:
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Ragu/ Minat menurut AI/ Excel", int(ai_mask.sum()))
            sc2.metric("Minat menurut Call Result (M1/M2/M3/Warm)", int(agent_mask.sum()))
            sc3.metric("Minat Aktual (Rekonsiliasi)", int(actual_mask.sum()))

    # =========================
    # TABS
    # =========================
    st.markdown("<hr style='margin: 25px 0 15px 0; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)

    if time_mode == "Bulanan":
        tab1, tab2, tab_hour, tab3 = st.tabs(["Overview", "Weekly Trend", "Hourly Trend", "Data & Detail"])
    else:
        tab1, tab_hour, tab3 = st.tabs(["Overview", "Hourly Trend", "Data & Detail"])
        tab2 = None

    # =========================
    # TAB 1: OVERVIEW
    # =========================
    with tab1:
        title = "Ringkasan Performa Bulanan per Aspek" if time_mode == "Bulanan" else "Ringkasan Performa Harian per Aspek"
        st.subheader(title)
        st.caption("Ditampilkan sebagai jumlah terpenuhi / total rekaman. Diurutkan dari aspek terlemah.")

        show_df = result_df.copy()
        if "_pct_main" in show_df.columns:
            show_df = show_df.drop(columns=["_pct_main"])

        show_df["Grade"] = show_df["Grade"].apply(grade_badge)

        weakest = set(show_df.head(5)["Aspek"].astype(str).tolist())
        styler = _highlight_rows_by_aspek(light_table(show_df), weakest)
        st.dataframe(styler, use_container_width=True, hide_index=True)

        if entity_type == "TL":
            st.write("")
            st.markdown("### Mitra yang Paling Menurunkan Performa TL")
            st.caption("Menampilkan agent terendah berdasarkan overall, beserta 4 aspek yang paling jarang disebutkan.")

            worst_df, best_df, _, _, _ = build_tl_agent_comparison_tables(
                scope_df=scope_df,
                aspect_cols=aspect_cols,
                top_n_aspects=4,
            )

            if worst_df is None or worst_df.empty:
                st.info("Tidak ada data agent yang cukup untuk dibandingkan pada periode ini.")
            else:
                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("#### Mitra Terburuk")
                    st.dataframe(light_table(worst_df), use_container_width=True, hide_index=True)

                with col_right:
                    st.markdown("#### Mitra Terbaik")
                    st.dataframe(light_table(best_df), use_container_width=True, hide_index=True)

    # =========================
    # TAB 2: WEEKLY TREND
    # =========================
    if time_mode == "Bulanan" and tab2 is not None:
        with tab2:
            st.subheader("Trend Overall per Minggu")

            yy, mm = selected_month.split("-")
            yy, mm = int(yy), int(mm)
            week_ranges = week_ranges_sun_sat_for_month(yy, mm)

            weekly_sel = []
            for i, (ws, we) in enumerate(week_ranges, start=1):
                df_w = scope_df[(scope_df[DATE_COL] >= ws) & (scope_df[DATE_COL] <= we)]
                overall_w = compute_overall_from_aspects(df_w, aspect_cols)
                weekly_sel.append({"Minggu": f"M{i}", "Overall": overall_w, "Jumlah Rekaman": len(df_w)})

            chart_df = pd.DataFrame(weekly_sel)

            base_month = baseline_df.copy()
            base_month["_month"] = base_month[DATE_COL].dt.to_period("M").astype(str)
            base_month = base_month[base_month["_month"] == selected_month].copy()

            min_line = max_line = kkm_line = np.nan
            if not base_month.empty and entity_col in base_month.columns:
                base_month[entity_col] = base_month[entity_col].astype(str).str.strip()
                base_month = base_month[(base_month[entity_col] != "") & (base_month[entity_col].str.lower() != "nan")]

                entity_week_vals = []
                for _, (ws, we) in enumerate(week_ranges, start=1):
                    d_week = base_month[(base_month[DATE_COL] >= ws) & (base_month[DATE_COL] <= we)]
                    if d_week.empty:
                        continue
                    for _, d_ent in d_week.groupby(entity_col):
                        ov = compute_overall_from_aspects(d_ent, aspect_cols)
                        if not np.isnan(ov):
                            entity_week_vals.append(float(ov))

                if entity_week_vals:
                    min_line = float(np.nanmin(entity_week_vals))
                    max_line = float(np.nanmax(entity_week_vals))
                    kkm_line = float(np.nanmean(entity_week_vals))

            df_kpi = chart_df.dropna(subset=["Overall"]).copy()
            avg_selected = float(df_kpi["Overall"].mean()) if not df_kpi.empty else np.nan

            base = alt.Chart(chart_df).encode(
                x=alt.X("Minggu:N", title="Minggu", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Overall:Q", title="Overall (%)"),
            )
            line = base.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=90)).encode(
                tooltip=[
                    alt.Tooltip("Minggu:N", title="Minggu"),
                    alt.Tooltip("Overall:Q", title="Overall (%)", format=".2f"),
                    alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                ]
            )
            area = base.mark_area(opacity=0.15)
            layers = [area, line]

            rule_df = []
            if not np.isnan(min_line):
                rule_df.append({"y": min_line, "label": "Min (Semua)"})
            if not np.isnan(max_line):
                rule_df.append({"y": max_line, "label": "Max (Semua)"})
            if not np.isnan(kkm_line):
                rule_df.append({"y": kkm_line, "label": "KKM (Semua)"})
            if not np.isnan(avg_selected):
                rule_df.append({"y": avg_selected, "label": f"Avg ({entity_type} terpilih)"})

            if rule_df:
                r = alt.Chart(pd.DataFrame(rule_df)).mark_rule(strokeDash=[6, 6]).encode(y="y:Q")
                t = alt.Chart(pd.DataFrame(rule_df)).mark_text(align="left", dx=6, dy=-6).encode(y="y:Q", text="label:N")
                layers.extend([r, t])

            st.altair_chart(style_chart(alt.layer(*layers), height=360), use_container_width=True)

            st.write("")
            st.caption("Volume WM per Minggu")

            if actual_mask is not None:
                weekly_wm = []
                for i, (ws, we) in enumerate(week_ranges, start=1):
                    df_w = scope_df[(scope_df[DATE_COL] >= ws) & (scope_df[DATE_COL] <= we)].copy()
                    wm = int(actual_mask.loc[df_w.index].sum()) if not df_w.empty else 0
                    weekly_wm.append({"Minggu": f"M{i}", "WM": wm})
                bar_df = pd.DataFrame(weekly_wm)
                bar_week = alt.Chart(bar_df).mark_bar().encode(
                    x=alt.X("Minggu:N", title="Minggu"),
                    y=alt.Y("WM:Q", title="WM"),
                    tooltip=[
                        alt.Tooltip("Minggu:N", title="Minggu"),
                        alt.Tooltip("WM:Q", title="WM"),
                    ],
                )
                st.altair_chart(style_chart(bar_week, height=160), use_container_width=True)
            else:
                bar_week = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X("Minggu:N", title="Minggu"),
                    y=alt.Y("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                    tooltip=[
                        alt.Tooltip("Minggu:N", title="Minggu"),
                        alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                    ],
                )
                st.altair_chart(style_chart(bar_week, height=160), use_container_width=True)

            st.write("")
            st.subheader("Jumlah Minat per Hari (Bulan Terpilih)")
            st.caption("Bar = jumlah minat, Line = rate minat (%).")

            if actual_mask is None:
                st.info("Tidak bisa buat grafik minat per hari karena kolom minat tidak tersedia.")
            else:
                dfd = scope_df.copy()
                dfd["_day_dt"] = pd.to_datetime(dfd[DATE_COL]).dt.floor("D")
                dfd["_is_interest"] = actual_mask.values

                daily = (
                    dfd.groupby("_day_dt", as_index=False)
                    .agg(jumlah_minat=("_is_interest", "sum"), jumlah_rekaman=("_is_interest", "size"))
                )
                daily["rate_minat"] = np.where(
                    daily["jumlah_rekaman"] > 0,
                    (daily["jumlah_minat"] / daily["jumlah_rekaman"] * 100.0),
                    np.nan,
                ).round(2)

                base_rate = alt.Chart(daily).encode(
                    x=alt.X("_day_dt:T", title="Tanggal", axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("rate_minat:Q", title="Rate Minat (%)"),
                    tooltip=[
                        alt.Tooltip("_day_dt:T", title="Tanggal"),
                        alt.Tooltip("jumlah_minat:Q", title="Jumlah Minat"),
                        alt.Tooltip("jumlah_rekaman:Q", title="Jumlah Rekaman"),
                        alt.Tooltip("rate_minat:Q", title="Rate Minat (%)", format=".2f"),
                    ],
                )
                st.altair_chart(
                    style_chart(
                        base_rate.mark_area(opacity=0.15) +
                        base_rate.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=90)),
                        height=260,
                    ),
                    use_container_width=True,
                )

                bar_cnt = alt.Chart(daily).mark_bar().encode(
                    x=alt.X("_day_dt:T", title="Tanggal", axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("jumlah_minat:Q", title="Jumlah Minat"),
                    tooltip=[
                        alt.Tooltip("_day_dt:T", title="Tanggal"),
                        alt.Tooltip("jumlah_minat:Q", title="Jumlah Minat"),
                        alt.Tooltip("jumlah_rekaman:Q", title="Jumlah Rekaman"),
                        alt.Tooltip("rate_minat:Q", title="Rate Minat (%)", format=".2f"),
                    ],
                )
                st.altair_chart(style_chart(bar_cnt, height=180), use_container_width=True)

            st.write("")
            st.subheader("Jumlah Minat per Hari per 2 Periode Bulan")
            st.caption("Grafik yang sama seperti 'Jumlah Minat per Hari (Bulan Terpilih)', tetapi dipisah menjadi 2 rentang tanggal dalam bulan kalender.")

            if actual_mask is None:
                st.info("Tidak bisa membuat grafik periode karena kolom minat tidak tersedia.")
            else:
                yy, mm = selected_month.split("-")
                yy, mm = int(yy), int(mm)

                period_info = split_calendar_month_ranges(yy, mm)

                daily_p1 = build_daily_interest_for_period(
                    scope_df=scope_df,
                    actual_mask=actual_mask,
                    start_day=period_info["p1_start"],
                    end_day=period_info["p1_end"],
                )

                daily_p2 = build_daily_interest_for_period(
                    scope_df=scope_df,
                    actual_mask=actual_mask,
                    start_day=period_info["p2_start"],
                    end_day=period_info["p2_end"],
                )

                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    st.markdown(f"#### {period_info['label_1']}")
                    if daily_p1 is None or daily_p1.empty:
                        st.info("Tidak ada data pada periode 1.")
                    else:
                        base_p1 = alt.Chart(daily_p1).encode(
                            x=alt.X("_day_dt:T", title="Tanggal", axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y("rate_minat:Q", title="Rate Minat (%)"),
                            tooltip=[
                                alt.Tooltip("_day_dt:T", title="Tanggal"),
                                alt.Tooltip("jumlah_minat:Q", title="Jumlah Minat"),
                                alt.Tooltip("jumlah_rekaman:Q", title="Jumlah Rekaman"),
                                alt.Tooltip("rate_minat:Q", title="Rate Minat (%)", format=".2f"),
                            ],
                        )
                        st.altair_chart(
                            style_chart(
                                base_p1.mark_area(opacity=0.15) +
                                base_p1.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=90)),
                                height=240,
                            ),
                            use_container_width=True,
                        )

                        bar_p1 = alt.Chart(daily_p1).mark_bar().encode(
                            x=alt.X("_day_dt:T", title="Tanggal", axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y("jumlah_minat:Q", title="Jumlah Minat"),
                            tooltip=[
                                alt.Tooltip("_day_dt:T", title="Tanggal"),
                                alt.Tooltip("jumlah_minat:Q", title="Jumlah Minat"),
                                alt.Tooltip("jumlah_rekaman:Q", title="Jumlah Rekaman"),
                                alt.Tooltip("rate_minat:Q", title="Rate Minat (%)", format=".2f"),
                            ],
                        )
                        st.altair_chart(style_chart(bar_p1, height=170), use_container_width=True)

                with col_p2:
                    st.markdown(f"#### {period_info['label_2']}")
                    if daily_p2 is None or daily_p2.empty:
                        st.info("Tidak ada data pada periode 2.")
                    else:
                        base_p2 = alt.Chart(daily_p2).encode(
                            x=alt.X("_day_dt:T", title="Tanggal", axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y("rate_minat:Q", title="Rate Minat (%)"),
                            tooltip=[
                                alt.Tooltip("_day_dt:T", title="Tanggal"),
                                alt.Tooltip("jumlah_minat:Q", title="Jumlah Minat"),
                                alt.Tooltip("jumlah_rekaman:Q", title="Jumlah Rekaman"),
                                alt.Tooltip("rate_minat:Q", title="Rate Minat (%)", format=".2f"),
                            ],
                        )
                        st.altair_chart(
                            style_chart(
                                base_p2.mark_area(opacity=0.15) +
                                base_p2.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=90)),
                                height=240,
                            ),
                            use_container_width=True,
                        )

                        bar_p2 = alt.Chart(daily_p2).mark_bar().encode(
                            x=alt.X("_day_dt:T", title="Tanggal", axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y("jumlah_minat:Q", title="Jumlah Minat"),
                            tooltip=[
                                alt.Tooltip("_day_dt:T", title="Tanggal"),
                                alt.Tooltip("jumlah_minat:Q", title="Jumlah Minat"),
                                alt.Tooltip("jumlah_rekaman:Q", title="Jumlah Rekaman"),
                                alt.Tooltip("rate_minat:Q", title="Rate Minat (%)", format=".2f"),
                            ],
                        )
                        st.altair_chart(style_chart(bar_p2, height=170), use_container_width=True)

            if entity_type == "TL":
                st.write("")
                st.subheader("Jumlah Minat per Hari (Agent Rutin)")
                st.caption("Menghitung agent yang kosong maksimal 2 hari selama bulan aktif.")

                if actual_mask is None:
                    st.info("Tidak bisa buat grafik ini karena kolom minat tidak tersedia.")
                else:
                    d = scope_df.copy()
                    d["_day_dt"] = pd.to_datetime(d[DATE_COL]).dt.floor("D")
                    active_days = sorted(d["_day_dt"].unique().tolist())

                    if not active_days:
                        st.info("Tidak ada hari aktif pada periode ini.")
                    else:
                        rutin_tbl = build_agent_daily_presence_summary(
                            scope_df=scope_df,
                            actual_mask=actual_mask,
                            active_days=active_days,
                            allowed_missing_days=2,
                            invert=False,
                        )

                        if rutin_tbl is None or rutin_tbl.empty:
                            st.info("Tidak ada agent yang memenuhi kriteria rutin (kosong maksimal 2 hari).")
                        else:
                            rutin_agents = set(rutin_tbl["Agent"].astype(str).tolist())
                            daily_rutin = build_daily_interest_chart_for_agent_group(
                                scope_df=scope_df,
                                actual_mask=actual_mask,
                                selected_agents=rutin_agents,
                            )

                            if daily_rutin is not None and not daily_rutin.empty:
                                base2 = alt.Chart(daily_rutin).encode(
                                    x=alt.X("_day_dt:T", title="Tanggal", axis=alt.Axis(labelAngle=-45)),
                                    y=alt.Y("rate_minat:Q", title="Rate Minat (%)"),
                                    tooltip=[
                                        alt.Tooltip("_day_dt:T", title="Tanggal"),
                                        alt.Tooltip("jumlah_minat:Q", title="Jumlah Minat"),
                                        alt.Tooltip("jumlah_rekaman:Q", title="Jumlah Rekaman"),
                                        alt.Tooltip("rate_minat:Q", title="Rate Minat (%)", format=".2f"),
                                    ],
                                )
                                st.altair_chart(
                                    style_chart(
                                        base2.mark_area(opacity=0.15) +
                                        base2.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=90)),
                                        height=260,
                                    ),
                                    use_container_width=True,
                                )

                                bar2 = alt.Chart(daily_rutin).mark_bar().encode(
                                    x=alt.X("_day_dt:T", title="Tanggal", axis=alt.Axis(labelAngle=-45)),
                                    y=alt.Y("jumlah_minat:Q", title="Jumlah Minat"),
                                    tooltip=[
                                        alt.Tooltip("_day_dt:T", title="Tanggal"),
                                        alt.Tooltip("jumlah_minat:Q", title="Jumlah Minat"),
                                        alt.Tooltip("jumlah_rekaman:Q", title="Jumlah Rekaman"),
                                        alt.Tooltip("rate_minat:Q", title="Rate Minat (%)", format=".2f"),
                                    ],
                                )
                                st.altair_chart(style_chart(bar2, height=180), use_container_width=True)
                                st.caption(f"Agent rutin terdeteksi: {len(rutin_agents)} agent.")

                            st.write("")
                            st.markdown("#### Tabel Agent Rutin (Kosong Maksimal 2 Hari)")
                            st.dataframe(light_table(rutin_tbl), use_container_width=True, hide_index=True)

                st.write("")
                st.subheader("Jumlah Minat per Hari (Agent Tidak Rutin)")
                st.caption("Kebalikan dari grafik sebelumnya: agent yang tidak masuk lebih dari 2 hari.")

                if actual_mask is None:
                    st.info("Tidak bisa buat grafik ini karena kolom minat tidak tersedia.")
                else:
                    d = scope_df.copy()
                    d["_day_dt"] = pd.to_datetime(d[DATE_COL]).dt.floor("D")
                    active_days = sorted(d["_day_dt"].unique().tolist())

                    if not active_days:
                        st.info("Tidak ada hari aktif pada periode ini.")
                    else:
                        tidak_rutin_tbl = build_agent_daily_presence_summary(
                            scope_df=scope_df,
                            actual_mask=actual_mask,
                            active_days=active_days,
                            allowed_missing_days=2,
                            invert=True,
                        )

                        if tidak_rutin_tbl is None or tidak_rutin_tbl.empty:
                            st.info("Tidak ada agent yang masuk kategori tidak rutin (> 2 hari kosong).")
                        else:
                            tidak_rutin_agents = set(tidak_rutin_tbl["Agent"].astype(str).tolist())
                            daily_tidak_rutin = build_daily_interest_chart_for_agent_group(
                                scope_df=scope_df,
                                actual_mask=actual_mask,
                                selected_agents=tidak_rutin_agents,
                            )

                            if daily_tidak_rutin is not None and not daily_tidak_rutin.empty:
                                base3 = alt.Chart(daily_tidak_rutin).encode(
                                    x=alt.X("_day_dt:T", title="Tanggal", axis=alt.Axis(labelAngle=-45)),
                                    y=alt.Y("rate_minat:Q", title="Rate Minat (%)"),
                                    tooltip=[
                                        alt.Tooltip("_day_dt:T", title="Tanggal"),
                                        alt.Tooltip("jumlah_minat:Q", title="Jumlah Minat"),
                                        alt.Tooltip("jumlah_rekaman:Q", title="Jumlah Rekaman"),
                                        alt.Tooltip("rate_minat:Q", title="Rate Minat (%)", format=".2f"),
                                    ],
                                )
                                st.altair_chart(
                                    style_chart(
                                        base3.mark_area(opacity=0.15) +
                                        base3.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=90)),
                                        height=260,
                                    ),
                                    use_container_width=True,
                                )

                                bar3 = alt.Chart(daily_tidak_rutin).mark_bar().encode(
                                    x=alt.X("_day_dt:T", title="Tanggal", axis=alt.Axis(labelAngle=-45)),
                                    y=alt.Y("jumlah_minat:Q", title="Jumlah Minat"),
                                    tooltip=[
                                        alt.Tooltip("_day_dt:T", title="Tanggal"),
                                        alt.Tooltip("jumlah_minat:Q", title="Jumlah Minat"),
                                        alt.Tooltip("jumlah_rekaman:Q", title="Jumlah Rekaman"),
                                        alt.Tooltip("rate_minat:Q", title="Rate Minat (%)", format=".2f"),
                                    ],
                                )
                                st.altair_chart(style_chart(bar3, height=180), use_container_width=True)
                                st.caption(f"Agent tidak rutin terdeteksi: {len(tidak_rutin_agents)} agent.")

                            st.write("")
                            st.markdown("#### Tabel Agent Tidak Rutin (> 2 Hari Kosong)")
                            st.dataframe(light_table(tidak_rutin_tbl), use_container_width=True, hide_index=True)

    # =========================
    # TAB: HOURLY TREND
    # =========================
    with tab_hour:
        st.subheader("Performa Overall per Jam (08:00–17:00)")
        st.caption("Jam 12:00–13:00 dianggap istirahat → dikosongkan. Skala grafik dipaksa dari 0 sampai 100.")

        if scope_df["_dt_call"].notna().sum() == 0:
            st.warning(f"Kolom `{DATETIME_COL}` tidak ditemukan/valid. Hourly Trend tidak bisa dihitung.")
        else:
            dfh = scope_df.dropna(subset=["_dt_call"]).copy()
            dfh["_hour"] = dfh["_dt_call"].dt.hour

            hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            active_hours = [8, 9, 10, 11, 13, 14, 15, 16, 17]
            sort_jam = [f"{h:02d}:00" for h in hours]

            hourly_rows = []
            for h in hours:
                if h == 12:
                    hourly_rows.append({"Jam": "12:00", "Overall": np.nan, "Jumlah Rekaman": np.nan, "Hour": 12})
                    continue

                d = dfh[dfh["_hour"] == h]
                if d.empty:
                    hourly_rows.append({"Jam": f"{h:02d}:00", "Overall": np.nan, "Jumlah Rekaman": np.nan, "Hour": h})
                    continue

                overall_h = compute_overall_from_aspects(d, aspect_cols)
                hourly_rows.append({
                    "Jam": f"{h:02d}:00",
                    "Overall": overall_h,
                    "Jumlah Rekaman": float(len(d)),
                    "Hour": h,
                })

            hour_df = pd.DataFrame(hourly_rows).sort_values("Hour")
            df_kpi = hour_df.dropna(subset=["Overall"]).copy()
            avg_overall = float(df_kpi["Overall"].mean()) if not df_kpi.empty else np.nan

            ref_lines = compute_hourly_reference_lines(
                baseline_df=baseline_df,
                aspect_cols=aspect_cols,
                entity_col=entity_col,
            )

            upper_line = ref_lines["upper"]
            lower_line = ref_lines["lower"]
            kkm_line = ref_lines["kkm"]

            base = alt.Chart(hour_df).encode(
                x=alt.X("Jam:N", sort=sort_jam, title="Jam", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(
                    "Overall:Q",
                    title="Overall (%)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(values=list(range(0, 101, 10))),
                ),
            )

            line = base.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=90)).encode(
                tooltip=[
                    alt.Tooltip("Jam:N", title="Jam"),
                    alt.Tooltip("Overall:Q", title="Overall (%)", format=".2f"),
                    alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                ]
            )
            area = base.mark_area(opacity=0.15)
            layers = [area, line]

            rule_df = []
            if not np.isnan(lower_line):
                rule_df.append({"y": lower_line, "label": "Min"})
            if not np.isnan(upper_line):
                rule_df.append({"y": upper_line, "label": "Max"})
            if not np.isnan(kkm_line):
                rule_df.append({"y": kkm_line, "label": "KKM"})
            if not np.isnan(avg_overall):
                rule_df.append({"y": avg_overall, "label": "Avg"})

            if rule_df:
                rule_source = pd.DataFrame(rule_df)
                r = alt.Chart(rule_source).mark_rule(strokeDash=[6, 6]).encode(y="y:Q")
                t = alt.Chart(rule_source).mark_text(align="left", dx=6, dy=-6).encode(y="y:Q", text="label:N")
                layers.extend([r, t])

            st.altair_chart(style_chart(alt.layer(*layers), height=340), use_container_width=True)

            st.write("")
            st.caption("Volume Rekaman per Jam")
            bar_vol = alt.Chart(hour_df).mark_bar().encode(
                x=alt.X("Jam:N", sort=sort_jam, title="Jam"),
                y=alt.Y("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                tooltip=[
                    alt.Tooltip("Jam:N", title="Jam"),
                    alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                ],
            )
            st.altair_chart(style_chart(bar_vol, height=160), use_container_width=True)

            st.write("")
            st.subheader("Jam Rawan Minat (Warm Leads / M1–M3)")
            st.caption("Bar = jumlah minat, Line = rate minat (%). Jam 12:00 dikosongkan.")

            if actual_mask is None:
                st.info("Tidak bisa buat grafik minat per jam karena kolom minat tidak tersedia.")
            else:
                dfhi = dfh.copy()
                dfhi = dfhi[dfhi["_hour"] != 12].copy()
                dfhi["_is_interest"] = actual_mask.loc[dfhi.index].values

                interest_rows = []
                for h in hours:
                    if h == 12:
                        interest_rows.append({
                            "Jam": "12:00",
                            "Jumlah Minat": np.nan,
                            "Jumlah Rekaman": np.nan,
                            "Rate Minat": np.nan,
                            "Hour": 12,
                        })
                        continue

                    d = dfhi[dfhi["_hour"] == h]
                    total = int(len(d))
                    minat = int(d["_is_interest"].sum())
                    rate = (minat / total * 100.0) if total > 0 else np.nan

                    interest_rows.append({
                        "Jam": f"{h:02d}:00",
                        "Jumlah Minat": (float(minat) if total > 0 else np.nan),
                        "Jumlah Rekaman": (float(total) if total > 0 else np.nan),
                        "Rate Minat": (round(float(rate), 2) if total > 0 else np.nan),
                        "Hour": h,
                    })

                hour_interest = pd.DataFrame(interest_rows).sort_values("Hour")

                base_r = alt.Chart(hour_interest).encode(
                    x=alt.X("Jam:N", sort=sort_jam, title="Jam", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(
                        "Rate Minat:Q",
                        title="Rate Minat (%)",
                        scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(values=list(range(0, 101, 10))),
                    ),
                    tooltip=[
                        alt.Tooltip("Jam:N", title="Jam"),
                        alt.Tooltip("Jumlah Minat:Q", title="Jumlah Minat"),
                        alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                        alt.Tooltip("Rate Minat:Q", title="Rate Minat (%)", format=".2f"),
                    ],
                )
                st.altair_chart(
                    style_chart(
                        base_r.mark_area(opacity=0.15) +
                        base_r.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=90)),
                        height=240,
                    ),
                    use_container_width=True,
                )

                bar_m = alt.Chart(hour_interest).mark_bar().encode(
                    x=alt.X("Jam:N", sort=sort_jam, title="Jam"),
                    y=alt.Y("Jumlah Minat:Q", title="Jumlah Minat"),
                    tooltip=[
                        alt.Tooltip("Jam:N", title="Jam"),
                        alt.Tooltip("Jumlah Minat:Q", title="Jumlah Minat"),
                        alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                        alt.Tooltip("Rate Minat:Q", title="Rate Minat (%)", format=".2f"),
                    ],
                )
                st.altair_chart(style_chart(bar_m, height=180), use_container_width=True)

                st.write("")
                st.subheader("Jam Rawan Tidak Minat")
                st.caption("Bar = jumlah tidak minat, Line = rate tidak minat (%). Jam 12:00 dikosongkan.")

                dfhn = dfh.copy()
                dfhn = dfhn[dfhn["_hour"] != 12].copy()
                dfhn["_is_not_interest"] = ~actual_mask.loc[dfhn.index].values

                not_interest_rows = []
                for h in hours:
                    if h == 12:
                        not_interest_rows.append({
                            "Jam": "12:00",
                            "Jumlah Tidak Minat": np.nan,
                            "Jumlah Rekaman": np.nan,
                            "Rate Tidak Minat": np.nan,
                            "Hour": 12,
                        })
                        continue

                    d = dfhn[dfhn["_hour"] == h]
                    total = int(len(d))
                    tidak_minat = int(d["_is_not_interest"].sum())
                    rate = (tidak_minat / total * 100.0) if total > 0 else np.nan

                    not_interest_rows.append({
                        "Jam": f"{h:02d}:00",
                        "Jumlah Tidak Minat": (float(tidak_minat) if total > 0 else np.nan),
                        "Jumlah Rekaman": (float(total) if total > 0 else np.nan),
                        "Rate Tidak Minat": (round(float(rate), 2) if total > 0 else np.nan),
                        "Hour": h,
                    })

                hour_not_interest = pd.DataFrame(not_interest_rows).sort_values("Hour")

                base_ni = alt.Chart(hour_not_interest).encode(
                    x=alt.X("Jam:N", sort=sort_jam, title="Jam", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(
                        "Rate Tidak Minat:Q",
                        title="Rate Tidak Minat (%)",
                        scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(values=list(range(0, 101, 10))),
                    ),
                    tooltip=[
                        alt.Tooltip("Jam:N", title="Jam"),
                        alt.Tooltip("Jumlah Tidak Minat:Q", title="Jumlah Tidak Minat"),
                        alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                        alt.Tooltip("Rate Tidak Minat:Q", title="Rate Tidak Minat (%)", format=".2f"),
                    ],
                )
                st.altair_chart(
                    style_chart(
                        base_ni.mark_area(opacity=0.15) +
                        base_ni.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=90)),
                        height=240,
                    ),
                    use_container_width=True,
                )

                bar_ni = alt.Chart(hour_not_interest).mark_bar().encode(
                    x=alt.X("Jam:N", sort=sort_jam, title="Jam"),
                    y=alt.Y("Jumlah Tidak Minat:Q", title="Jumlah Tidak Minat"),
                    tooltip=[
                        alt.Tooltip("Jam:N", title="Jam"),
                        alt.Tooltip("Jumlah Tidak Minat:Q", title="Jumlah Tidak Minat"),
                        alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                        alt.Tooltip("Rate Tidak Minat:Q", title="Rate Tidak Minat (%)", format=".2f"),
                    ],
                )
                st.altair_chart(style_chart(bar_ni, height=180), use_container_width=True)

            st.write("")
            st.subheader("Jam Diangkat Berdasarkan Jenis Call")
            st.caption("Menampilkan jumlah call per jam untuk kategori M1, M2, M3, dan Tidak Minat.")

            if CALLRESULT_COL not in dfh.columns:
                st.info("Kolom call result tidak tersedia, sehingga grafik jenis call per jam tidak bisa dibuat.")
            else:
                df_call = dfh.copy()
                df_call = df_call[df_call["_hour"] != 12].copy()
                df_call["Jenis Call"] = df_call[CALLRESULT_COL].apply(normalize_call_bucket)
                df_call = df_call[df_call["Jenis Call"].notna()].copy()

                if df_call.empty:
                    st.info("Tidak ada data M1/M2/M3/Tidak Minat pada periode ini.")
                else:
                    base_grid = pd.MultiIndex.from_product(
                        [active_hours, ["M1", "M2", "M3", "Tidak Minat"]],
                        names=["Hour", "Jenis Call"],
                    ).to_frame(index=False)

                    call_hour = (
                        df_call.groupby(["_hour", "Jenis Call"], as_index=False)
                        .size()
                        .rename(columns={"_hour": "Hour", "size": "Jumlah Rekaman"})
                    )

                    call_hour = base_grid.merge(call_hour, on=["Hour", "Jenis Call"], how="left")
                    call_hour["Jumlah Rekaman"] = call_hour["Jumlah Rekaman"].fillna(0)
                    call_hour["Jam"] = call_hour["Hour"].map(lambda x: f"{int(x):02d}:00")

                    chart_call = alt.Chart(call_hour).mark_bar().encode(
                        x=alt.X("Jam:N", sort=sort_jam, title="Jam"),
                        y=alt.Y("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                        color=alt.Color("Jenis Call:N", title="Jenis Call"),
                        tooltip=[
                            alt.Tooltip("Jam:N", title="Jam"),
                            alt.Tooltip("Jenis Call:N", title="Jenis Call"),
                            alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                        ],
                    )
                    st.altair_chart(style_chart(chart_call, height=260), use_container_width=True)

            st.write("")
            st.subheader("Durasi Rata-Rata Tiap Jam")
            st.caption("Kolom duration dibaca sebagai detik. Contoh 40 = 40 detik, 120 = 2 menit, 95 = 1 menit 35 detik.")

            if DURATION_COL not in dfh.columns:
                st.info(f"Kolom `{DURATION_COL}` tidak ditemukan, sehingga grafik durasi rata-rata per jam tidak bisa dibuat.")
            else:
                dfdur = dfh.copy()
                dfdur = dfdur[dfdur["_hour"] != 12].copy()
                dfdur["_duration_sec"] = dfdur[DURATION_COL].apply(parse_duration_seconds)

                duration_rows = []
                for h in hours:
                    if h == 12:
                        duration_rows.append({
                            "Jam": "12:00",
                            "Durasi Rata-rata (detik)": np.nan,
                            "Durasi Rata-rata": None,
                            "Jumlah Rekaman Valid": np.nan,
                            "Hour": 12,
                        })
                        continue

                    d = dfdur[dfdur["_hour"] == h].dropna(subset=["_duration_sec"])
                    if d.empty:
                        duration_rows.append({
                            "Jam": f"{h:02d}:00",
                            "Durasi Rata-rata (detik)": np.nan,
                            "Durasi Rata-rata": None,
                            "Jumlah Rekaman Valid": np.nan,
                            "Hour": h,
                        })
                        continue

                    avg_sec = float(d["_duration_sec"].mean())
                    duration_rows.append({
                        "Jam": f"{h:02d}:00",
                        "Durasi Rata-rata (detik)": round(avg_sec, 2),
                        "Durasi Rata-rata": format_seconds_mmss(avg_sec),
                        "Jumlah Rekaman Valid": int(len(d)),
                        "Hour": h,
                    })

                duration_df = pd.DataFrame(duration_rows).sort_values("Hour")

                base_dur = alt.Chart(duration_df).encode(
                    x=alt.X("Jam:N", sort=sort_jam, title="Jam", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Durasi Rata-rata (detik):Q", title="Durasi Rata-rata (detik)"),
                    tooltip=[
                        alt.Tooltip("Jam:N", title="Jam"),
                        alt.Tooltip("Durasi Rata-rata:N", title="Durasi Rata-rata"),
                        alt.Tooltip("Durasi Rata-rata (detik):Q", title="Detik", format=".2f"),
                        alt.Tooltip("Jumlah Rekaman Valid:Q", title="Jumlah Rekaman Valid"),
                    ],
                )

                chart_dur = base_dur.mark_area(opacity=0.15) + base_dur.mark_line(
                    strokeWidth=4,
                    point=alt.OverlayMarkDef(size=90),
                )
                st.altair_chart(style_chart(chart_dur, height=260), use_container_width=True)

    # =========================
    # TAB: DATA & DETAIL
    # =========================
    with tab3:
        st.subheader("Data & Detail")

        if show_agent_interest_table_in_tab3:
            st.markdown("### Ringkasan Agent (Minat vs Tidak Minat)")
            st.caption("Mengikuti filter call type + periode yang sedang dipilih (harian/bulanan).")

            agent_summary = build_agent_interest_summary(scope_df)
            if agent_summary is None:
                st.info(
                    "Tidak bisa tampilkan ringkasan minat per Agent karena kolom minat tidak tersedia. "
                    f"Butuh: `{SENTIMENT_CATEGORY_COL}`+`{SENTIMENT_REASON_COL}` atau `{LOV3_COL}`, dan `{CALLRESULT_COL}`."
                )
            else:
                st.dataframe(light_table(agent_summary), use_container_width=True, hide_index=True)

            st.markdown("<hr style='margin: 14px 0; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)

        t1, t2 = build_priority_followup_tables(scope_df)

        st.markdown("### Data Call Minat dimana Agent yang Jarang Menyebutkan 5 Aspek Terlemah")
        st.caption(
            "Menampilkan call bertipe M1, M2, atau M3 yang masih belum / jarang menyebut aspek penting: "
            "Menyebut Choice Customer, Menyebut Termasuk Angsuran, Segmentation Offer Range, "
            "Menyebut Benefit, dan Melakukan Simulasi."
        )
        if t1 is None or t1.empty:
            st.info("Tidak ada data M1/M2/M3 yang memenuhi kriteria tabel 1 pada periode ini.")
        else:
            st.dataframe(light_table(t1), use_container_width=True, hide_index=True)

        st.markdown("<hr style='margin: 14px 0; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)

        st.markdown("### Data Call Tidak Minat dimana Agent sudah melakukan simulasi yang dapat dijadikan Follow Up")
        st.caption(
            "Menampilkan call bertipe Tidak Minat yang ternyata sudah menyebut salah satu / beberapa aspek berikut: "
            "Menyebut Choice Customer, Menyebut Termasuk Angsuran, dan Melakukan Simulasi."
        )
        if t2 is None or t2.empty:
            st.info("Tidak ada data Tidak Minat yang memenuhi kriteria tabel 2 pada periode ini.")
        else:
            st.dataframe(light_table(t2), use_container_width=True, hide_index=True)

        st.markdown("<hr style='margin: 14px 0; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)

        st.subheader("Detail Data untuk Scoring")
        st.caption("Menampilkan 50 baris pertama sesuai filter periode aktif.")
        st.dataframe(light_table(scope_df.head(50)), use_container_width=True, hide_index=True)

# =========================
# SIDEBAR: UPLOAD + LOGO
# =========================
with st.sidebar:
    st.markdown("### 🎛️ Filter & Data")
    uploaded = st.file_uploader("Upload file QC (CSV/XLSX)", type=["csv", "xlsx", "xls"])

st.sidebar.markdown(
    """
    <div class="sidebar-logo">
        <img src="https://oss-api.berijalan.id/web-berijalan/img/logo-berijalan.webp" width="150"/>
        <p>@ QC Audio Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# MAIN HEADER
# =========================
st.markdown(
    """
    <div class="hero">
      <p class="hero-title">QC Audio Dashboard</p>
      <div class="hero-sub">
        Monitoring performa Mitra (Agent) dan Team Leader (TL) otomatis (harian jika 1 tanggal, bulanan jika >1 tanggal),
        per jam (08–17; 12:00 istirahat), ringkasan performa per aspek, serta grafik jam/tangga
        l rawan minat.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

if uploaded is None:
    st.info("Upload file QC dulu di sidebar untuk mulai.")
    st.stop()

# =========================
# SHOW LOADER DURING PYTHON PROCESS
# =========================
processing_flag = st.empty()
processing_flag.markdown('<div id="processing-flag"></div>', unsafe_allow_html=True)

try:
    df = load_data(uploaded)
except Exception as e:
    processing_flag.empty()
    st.error(f"Gagal baca file: {e}")
    st.stop()

missing = [c for c in [TL_COL, AGENT_COL] if c not in df.columns]
if missing:
    processing_flag.empty()
    st.error(f"Kolom wajib tidak ditemukan: {missing}\n\nKolom yang ada: {list(df.columns)}")
    st.stop()

df_clean = normalize_identity_cols(df)
processing_flag.empty()

# =========================
# SIDEBAR: KATEGORI PENILAIAN
# =========================
with st.sidebar:
    st.markdown("### Kategori Penilaian")
    mode = st.radio("Pilih yang dinilai:", ["Agent", "TL"], horizontal=True, key="mode_penilaian")

tl_list = sorted(df_clean[TL_COL].unique().tolist())

# =========================
# RENDER MAIN
# =========================
if mode == "Agent":
    with st.sidebar:
        st.markdown("### 👤 Penilaian Agent")
        selected_tl = st.selectbox("Pilih Team Leader (TL)", tl_list, key="tl_agent")

    df_tl_all = df.copy()
    df_tl_all[TL_COL] = df_tl_all[TL_COL].astype(str).str.strip()
    df_tl_all[AGENT_COL] = df_tl_all[AGENT_COL].astype(str).str.strip()
    df_tl_all = df_tl_all[df_tl_all[TL_COL] == selected_tl].copy()

    agent_list = sorted(df_clean.loc[df_clean[TL_COL] == selected_tl, AGENT_COL].unique().tolist())
    with st.sidebar:
        selected_agent = st.selectbox("Pilih Mitra (Agent)", agent_list, key="agent")

    df_sel = df.copy()
    df_sel[TL_COL] = df_sel[TL_COL].astype(str).str.strip()
    df_sel[AGENT_COL] = df_sel[AGENT_COL].astype(str).str.strip()
    df_sel = df_sel[(df_sel[TL_COL] == selected_tl) & (df_sel[AGENT_COL] == selected_agent)].copy()

    if df_sel.empty:
        st.warning("Tidak ada data untuk TL+Agent ini.")
        st.stop()

    header_badges = f"""
<span class="badge">TL: <b>{selected_tl}</b></span>
&nbsp; <span class="badge">Mitra: <b>{selected_agent}</b></span>
"""
    run_performance_block(
        df_base=df_sel,
        header_badges_html=header_badges,
        title_context="Mitra yang terpilih:",
        key_prefix="agent_view",
        show_agent_interest_table_in_tab3=False,
        baseline_df=df_tl_all,
    )

else:
    with st.sidebar:
        st.markdown("### 👥 Penilaian TL")
        selected_tl = st.selectbox("Pilih Team Leader (TL)", tl_list, key="tl_only")

    df_sel = df.copy()
    df_sel[TL_COL] = df_sel[TL_COL].astype(str).str.strip()
    df_sel[AGENT_COL] = df_sel[AGENT_COL].astype(str).str.strip()
    df_sel = df_sel[df_sel[TL_COL] == selected_tl].copy()

    if df_sel.empty:
        st.warning("Tidak ada data untuk TL ini.")
        st.stop()

    agent_count = (
        df_sel[AGENT_COL].astype(str).str.strip()
        .replace("nan", "").replace("", np.nan).dropna().nunique()
    )

    header_badges = f"""
<span class="badge">TL: <b>{selected_tl}</b></span>
&nbsp; <span class="badge">Total Agent: <b>{agent_count}</b></span>
"""

    df_all_tl = df.copy()
    df_all_tl[TL_COL] = df_all_tl[TL_COL].astype(str).str.strip()
    df_all_tl = df_all_tl[(df_all_tl[TL_COL] != "") & (df_all_tl[TL_COL].str.lower() != "nan")].copy()

    run_performance_block(
        df_base=df_sel,
        header_badges_html=header_badges,
        title_context="Team Leader yang dipilih:",
        key_prefix="tl_view",
        show_agent_interest_table_in_tab3=True,
        baseline_df=df_all_tl,
    )

st.write("")
st.caption("© QC Audio Dashboard")

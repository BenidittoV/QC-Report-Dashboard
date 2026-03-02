import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import json

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

SENTIMENT_CATEGORY_COL = "sentiment_category"
SENTIMENT_REASON_COL = "sentiment_reason"

# fallback jika sentiment_* tidak ada
LOV3_COL = "metadata_resultLov3"

DEFAULT_ALLOWED_CALL_TYPES = [
    "M1 (Setuju dikirim hitungan)", "M2 (Negosiasi)", "M3 (Setuju dengan hitungan)",
    "Tidak Minat", "Warm Leads", "Pencairan Minus"
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

LOADING_IMAGE_URL = "https://static.vecteezy.com/system/resources/previews/010/754/321/non_2x/loading-icon-logo-design-template-illustration-free-vector.jpg"

# =========================
# MODERN CSS + LOADING OVERLAY
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

/* =========================
   LOADING OVERLAY (FULLSCREEN)
   ========================= */
@keyframes spin {{
  0%   {{ transform: rotate(0deg); }}
  100% {{ transform: rotate(360deg); }}
}}
#loading-overlay {{
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(255,255,255,0.92);
  z-index: 999999;
  display: flex;
  align-items: center;
  justify-content: center;
}}
#loading-overlay .loading-inner {{
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 18px 22px;
  border-radius: 18px;
  border: 1px solid #e5e7eb;
  background: rgba(255,255,255,0.9);
  box-shadow: 0 18px 40px rgba(0,0,0,0.10);
}}
#loading-overlay img.loading-logo {{
  width: 110px;
  height: 110px;
  border-radius: 999px;
  animation: spin 1.0s linear infinite;
}}
#loading-overlay .loading-text {{
  font-weight: 800;
  color: #111827;
  opacity: 0.9;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

LOADING_OVERLAY_HTML = f"""
<div id="loading-overlay">
  <div class="loading-inner">
    <img class="loading-logo" src="{LOADING_IMAGE_URL}" />
    <div class="loading-text">Memproses file…</div>
  </div>
</div>
"""

def show_loading_overlay():
    ph = st.empty()
    ph.markdown(LOADING_OVERLAY_HTML, unsafe_allow_html=True)
    return ph

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
    if p >= 90: return "S"
    if p >= 80: return "A"
    if p >= 70: return "B"
    if p >= 60: return "C"
    return "D"

def safe_pct(binary_series: pd.Series) -> float:
    if binary_series is None or len(binary_series) == 0:
        return np.nan
    return round(float(binary_series.mean() * 100.0), 2)

def grade_badge(grade: str) -> str:
    if grade == "S": return "🏆 S"
    if grade == "A": return "✅ A"
    if grade == "B": return "👍 B"
    if grade == "C": return "⚠️ C"
    if grade == "D": return "⛔ D"
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
    month_end = (month_start + pd.offsets.MonthEnd(1))
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
    num_cols = df.select_dtypes(include=[np.number]).columns
    fmt = {c: "{:.2f}" for c in num_cols}
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
    s_low = s_low.str.replace(r"^(\d{{1,2}})\s+(\d{{2}})\s+(\d{{4}})\s+", r"\1-\2-\3 ", regex=True)
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
            return pd.DataFrame()

    out = out.dropna(subset=[DATE_COL])
    if out.empty:
        st.warning("Semua tanggal gagal diparse. Pastikan format tanggal valid.")
        return pd.DataFrame()

    return out

def compute_overall_from_aspects(df_subset: pd.DataFrame, aspect_cols: list[str]) -> float:
    if df_subset.empty:
        return np.nan
    vals = [safe_pct(df_subset[c].apply(normalize_to_binary)) for c in aspect_cols]
    return round(float(np.nanmean(vals)), 2) if len(vals) else np.nan

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
            (~agent_mask) &
            (dfx[SENTIMENT_CATEGORY_COL] == "tidak berminat") &
            (dfx[SENTIMENT_REASON_COL].str.contains(pattern, na=False))
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

# ===== chart helpers =====
def style_chart(chart, height: int):
    return (
        chart.properties(height=height, background="white")
        .configure_view(stroke=None, fill="white")
        .configure_axis(labelColor="black", titleColor="black", gridColor="#e5e7eb")
    )

# =========================
# CORE RENDER BLOCK
# =========================
def run_performance_block(df_base: pd.DataFrame, header_badges_html: str, title_context: str, key_prefix: str):
    df_base = filter_call_types(df_base)
    if df_base.empty:
        st.warning("Tidak ada data setelah filter call type.")
        return

    df_base = ensure_date_and_dt(df_base)
    if df_base.empty:
        return

    # AUTO DETECT MODE:
    unique_days = int(df_base[DATE_COL].dt.date.nunique())
    time_mode = "Harian" if unique_days <= 1 else "Bulanan"

    # aspek
    aspect_cols = [c for c in ASPECT_COLUMNS_CANDIDATES if c in df_base.columns]
    missing_aspects = [c for c in ASPECT_COLUMNS_CANDIDATES if c not in df_base.columns]
    if not aspect_cols:
        st.error("Tidak ada kolom aspek yang ditemukan di file untuk dihitung.")
        return

    selected_period_label = ""
    selected_month = None

    # =========================
    # SELECT PERIOD (NOT IN SIDEBAR)
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
        selected_period_label = f"{selected_month}"

        dfm = df_base[df_base["_month"] == selected_month].copy()
        if dfm.empty:
            st.warning("Tidak ada data untuk bulan terpilih.")
            return

        yy, mm = selected_month.split("-")
        yy, mm = int(yy), int(mm)
        week_ranges = week_ranges_sun_sat_for_month(yy, mm)

        rows = []
        for col in aspect_cols:
            row = {"Aspek": ASPECT_FRIENDLY_NAMES.get(col, col)}
            for i, (ws, we) in enumerate(week_ranges, start=1):
                df_w = dfm[(dfm[DATE_COL] >= ws) & (dfm[DATE_COL] <= we)]
                pct_w = safe_pct(df_w[col].apply(normalize_to_binary))
                row[f"Minggu {i} (%)"] = pct_w if not np.isnan(pct_w) else np.nan

            pct_m = safe_pct(dfm[col].apply(normalize_to_binary))
            row["Persentase Bulanan (%)"] = pct_m if not np.isnan(pct_m) else np.nan
            row["Grade"] = tier_from_percent(pct_m) if not np.isnan(pct_m) else "-"
            rows.append(row)

        result_df = pd.DataFrame(rows).sort_values(by="Persentase Bulanan (%)", ascending=True)
        overall_value = round(float(pd.to_numeric(result_df["Persentase Bulanan (%)"], errors="coerce").mean()), 2)
        scope_df = dfm

    else:
        if df_base["_dt_call"].notna().sum() == 0:
            st.error(f"Mode Harian membutuhkan `{DATETIME_COL}` yang valid (contoh: 19 Februari 2026 08:07:05).")
            return

        only_day = df_base[DATE_COL].dt.date.unique()
        only_day = only_day[0] if len(only_day) else "-"
        selected_period_label = str(only_day)

        dfm = df_base.copy()

        buckets = [
            ("08–10 (%)", 8, 10),
            ("10–12 (%)", 10, 12),
            ("13–15 (%)", 13, 15),
            ("15–17 (%)", 15, 17),
        ]

        dfm["_hour"] = dfm["_dt_call"].dt.hour
        outside_mask = (dfm["_hour"] < 8) | (dfm["_hour"] >= 17)
        has_outside = bool(outside_mask.any())

        rows = []
        for col in aspect_cols:
            row = {"Aspek": ASPECT_FRIENDLY_NAMES.get(col, col)}
            for label, h0, h1 in buckets:
                d = dfm[(dfm["_hour"] >= h0) & (dfm["_hour"] < h1)]
                pct = safe_pct(d[col].apply(normalize_to_binary))
                row[label] = pct if not np.isnan(pct) else np.nan

            if has_outside:
                d_out = dfm[outside_mask]
                pct_out = safe_pct(d_out[col].apply(normalize_to_binary))
                row["Di luar 08–17 (%)"] = pct_out if not np.isnan(pct_out) else np.nan

            pct_d = safe_pct(dfm[col].apply(normalize_to_binary))
            row["Persentase Harian (%)"] = pct_d if not np.isnan(pct_d) else np.nan
            row["Grade"] = tier_from_percent(pct_d) if not np.isnan(pct_d) else "-"
            rows.append(row)

        result_df = pd.DataFrame(rows).sort_values(by="Persentase Harian (%)", ascending=True)
        overall_value = round(float(pd.to_numeric(result_df["Persentase Harian (%)"], errors="coerce").mean()), 2)
        scope_df = dfm

    # =========================
    # KPI HEADER (kotak mitra) + period badge
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
            unsafe_allow_html=True
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
        st.caption("Diurutkan dari yang terlemah ke terkuat.")

        show_df = result_df.copy()
        show_df["Grade"] = show_df["Grade"].apply(grade_badge)

        pct_cols = [c for c in show_df.columns if "(%)" in c]
        show_df_display = show_df.copy()
        for c in pct_cols:
            show_df_display[c] = show_df_display[c].apply(lambda x: "Tidak ada rekaman" if pd.isna(x) else f"{x:.2f}%")

        st.dataframe(light_table(show_df_display), use_container_width=True, hide_index=True)

        colL, colR = st.columns(2)
        key_pct = "Persentase Bulanan (%)" if time_mode == "Bulanan" else "Persentase Harian (%)"
        with colL:
            st.markdown("### Top 5 Aspek Terlemah")
            st.dataframe(light_table(show_df_display.head(5)[["Aspek", key_pct, "Grade"]]), use_container_width=True, hide_index=True)
        with colR:
            st.markdown("### Top 5 Aspek Terkuat")
            st.dataframe(
                light_table(show_df_display.sort_values(by=key_pct, ascending=False).head(5)[["Aspek", key_pct, "Grade"]]),
                use_container_width=True,
                hide_index=True
            )

    # =========================
    # TAB 2: WEEKLY TREND (ONLY BULANAN)
    # =========================
    if time_mode == "Bulanan" and tab2 is not None:
        with tab2:
            st.subheader("Trend Overall per Minggu")

            yy, mm = selected_month.split("-")
            yy, mm = int(yy), int(mm)
            week_ranges = week_ranges_sun_sat_for_month(yy, mm)

            weekly = []
            for i, (ws, we) in enumerate(week_ranges, start=1):
                df_w = scope_df[(scope_df[DATE_COL] >= ws) & (scope_df[DATE_COL] <= we)]
                overall_w = compute_overall_from_aspects(df_w, aspect_cols)
                weekly.append({"Minggu": f"M{i}", "Overall": overall_w, "Jumlah Rekaman": len(df_w)})

            chart_df = pd.DataFrame(weekly)
            df_kpi = chart_df.dropna(subset=["Overall"]).copy()
            avg_weekly = float(df_kpi["Overall"].mean()) if not df_kpi.empty else np.nan

            base = alt.Chart(chart_df).encode(
                x=alt.X("Minggu:N", title="Minggu", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Overall:Q", title="Overall (%)")
            )
            line = base.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=90)).encode(
                tooltip=[
                    alt.Tooltip("Minggu:N", title="Minggu"),
                    alt.Tooltip("Overall:Q", title="Overall (%)", format=".2f"),
                    alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                ]
            )
            area = base.mark_area(opacity=0.15)

            if not np.isnan(avg_weekly):
                rule = alt.Chart(pd.DataFrame({"avg": [avg_weekly]})).mark_rule(strokeDash=[6, 6]).encode(y="avg:Q")
                label = alt.Chart(pd.DataFrame({"avg": [avg_weekly], "txt": [f"Avg {avg_weekly:.1f}%"]})).mark_text(
                    align="left", dx=6, dy=-6
                ).encode(y="avg:Q", text="txt:N")
                chart_overall = style_chart(area + line + rule + label, height=360)
            else:
                chart_overall = style_chart(area + line, height=360)

            st.altair_chart(chart_overall, use_container_width=True)

            st.write("")
            st.caption("Volume Rekaman per Minggu")
            bar_week = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Minggu:N", title="Minggu"),
                y=alt.Y("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                tooltip=[alt.Tooltip("Minggu:N", title="Minggu"), alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman")]
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
                    np.nan
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
                line_rate = base_rate.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=90))
                area_rate = base_rate.mark_area(opacity=0.15)
                st.altair_chart(style_chart(area_rate + line_rate, height=260), use_container_width=True)

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

    # =========================
    # TAB: HOURLY TREND
    # =========================
    with tab_hour:
        st.subheader("Performa Overall per Jam (08:00–17:00)")
        st.caption("Jam 12:00–13:00 dianggap istirahat → dikosongkan.")

        if scope_df["_dt_call"].notna().sum() == 0:
            st.warning(f"Kolom `{DATETIME_COL}` tidak ditemukan/valid. Hourly Trend tidak bisa dihitung.")
        else:
            dfh = scope_df.dropna(subset=["_dt_call"]).copy()
            dfh["_hour"] = dfh["_dt_call"].dt.hour

            hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
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
                hourly_rows.append({"Jam": f"{h:02d}:00", "Overall": overall_h, "Jumlah Rekaman": float(len(d)), "Hour": h})

            hour_df = pd.DataFrame(hourly_rows).sort_values("Hour")
            df_kpi = hour_df.dropna(subset=["Overall"]).copy()
            avg_overall = float(df_kpi["Overall"].mean()) if not df_kpi.empty else np.nan

            base = alt.Chart(hour_df).encode(
                x=alt.X("Jam:N", sort=sort_jam, title="Jam", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Overall:Q", title="Overall (%)"),
            )
            line = base.mark_line(point=alt.OverlayMarkDef(size=80), strokeWidth=4).encode(
                tooltip=[
                    alt.Tooltip("Jam:N", title="Jam"),
                    alt.Tooltip("Overall:Q", title="Overall (%)", format=".2f"),
                    alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                ]
            )
            area = base.mark_area(opacity=0.18)

            if not np.isnan(avg_overall):
                rule = alt.Chart(pd.DataFrame({"avg": [avg_overall]})).mark_rule(strokeDash=[6, 6]).encode(y="avg:Q")
                label = alt.Chart(pd.DataFrame({"avg": [avg_overall], "txt": [f"Avg {avg_overall:.1f}%"]})).mark_text(
                    align="left", dx=6, dy=-6
                ).encode(y="avg:Q", text="txt:N")
                chart_hourly = style_chart(area + line + rule + label, height=320)
            else:
                chart_hourly = style_chart(area + line, height=320)

            st.altair_chart(chart_hourly, use_container_width=True)

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

            im2 = interest_masks(scope_df)
            actual_mask2 = im2["actual_mask"] if im2 is not None else None

            if actual_mask2 is None:
                st.info("Tidak bisa buat grafik minat per jam karena kolom minat tidak tersedia.")
            else:
                dfhi = dfh.copy()
                dfhi = dfhi[dfhi["_hour"] != 12].copy()
                dfhi["_is_interest"] = actual_mask2.loc[dfhi.index].values

                interest_rows = []
                for h in hours:
                    if h == 12:
                        interest_rows.append({"Jam": "12:00", "Jumlah Minat": np.nan, "Jumlah Rekaman": np.nan, "Rate Minat": np.nan, "Hour": 12})
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
                        "Hour": h
                    })

                hour_interest = pd.DataFrame(interest_rows).sort_values("Hour")

                base_r = alt.Chart(hour_interest).encode(
                    x=alt.X("Jam:N", sort=sort_jam, title="Jam", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Rate Minat:Q", title="Rate Minat (%)"),
                    tooltip=[
                        alt.Tooltip("Jam:N", title="Jam"),
                        alt.Tooltip("Jumlah Minat:Q", title="Jumlah Minat"),
                        alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                        alt.Tooltip("Rate Minat:Q", title="Rate Minat (%)", format=".2f"),
                    ],
                )
                line_r = base_r.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=90))
                area_r = base_r.mark_area(opacity=0.15)
                st.altair_chart(style_chart(area_r + line_r, height=240), use_container_width=True)

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

    # =========================
    # TAB: DATA & DETAIL
    # =========================
    with tab3:
        st.subheader("Detail Data untuk Scoring")
        st.caption("Tabel minat actual customer dihapus (tidak ada data customer).")
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
    unsafe_allow_html=True
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
        per jam (08–17; 12:00 istirahat), ringkasan performa per aspek, serta grafik jam/tanggal rawan minat.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

# =========================
# GATE: WAIT UPLOAD
# =========================
if uploaded is None:
    st.info("Upload file QC dulu di sidebar untuk mulai.")
    st.stop()

# =========================
# SHOW FULLSCREEN LOADER WHILE READING + PREP
# =========================
loader = show_loading_overlay()

try:
    df = load_data(uploaded)
except Exception as e:
    loader.empty()
    st.error(f"Gagal baca file: {e}")
    st.stop()

missing = [c for c in [TL_COL, AGENT_COL] if c not in df.columns]
if missing:
    loader.empty()
    st.error(f"Kolom wajib tidak ditemukan: {missing}\n\nKolom yang ada: {list(df.columns)}")
    st.stop()

df_clean = normalize_identity_cols(df)

# =========================
# SIDEBAR: KATEGORI PENILAIAN (ONLY)
# =========================
with st.sidebar:
    st.markdown("### Kategori Penilaian")
    mode = st.radio("Pilih yang dinilai:", ["Agent", "TL"], horizontal=True, key="mode_penilaian")

tl_list = sorted(df_clean[TL_COL].unique().tolist())

# selesai fase "memproses file" (loader ditutup sebelum render utama)
loader.empty()

# =========================
# RENDER MAIN
# =========================
if mode == "Agent":
    with st.sidebar:
        st.markdown("### 👤 Penilaian Agent")
        selected_tl = st.selectbox("Pilih Team Leader (TL)", tl_list, key="tl_agent")

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
        key_prefix="agent_view"
    )

else:
    with st.sidebar:
        st.markdown("### 👥 Penilaian TL")
        selected_tl = st.selectbox("Pilih Team Leader (TL)", tl_list, key="tl_only")

    df_sel = df.copy()
    df_sel[TL_COL] = df_sel[TL_COL].astype(str).str.strip()
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
    run_performance_block(
        df_base=df_sel,
        header_badges_html=header_badges,
        title_context="Team Leader yang dipilih:",
        key_prefix="tl_view"
    )

st.write("")
st.caption("© QC Audio Dashboard")

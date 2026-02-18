import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

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
DATE_COL = "call_date"  # format YYYY-MM-DD
DATETIME_COL = "metadata_dateCall"  # contoh: "01 November 2025 08:23:18"

DEFAULT_ALLOWED_CALL_TYPES = [
    "M1 (Setuju dikirim hitungan)", "M2 (Negosiasi)", "M3 (Setuju dengan hitungan)"
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

# =========================
# MODERN CSS (WHITE BG + BLUE SIDEBAR + MAIN TEXT BLACK) + FORCE LIGHT VEGA
# =========================
CUSTOM_CSS = """
<style>
html, body, [class*="css"]{
  font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont;
}
.stApp, .stApp * { color: #000000; }
.stApp { background-color: #ffffff; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #002D72, #002D72);
  color: white !important;
  border-right: none;
}
section[data-testid="stSidebar"], 
section[data-testid="stSidebar"] * {
  color: white !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea {
  background-color: rgba(255,255,255,0.14) !important;
  color: white !important;
  border-radius: 10px !important;
  border: none !important;
}
section[data-testid="stSidebar"] .stCaption, 
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: rgba(255,255,255,0.88) !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{
  background: rgba(255,255,255,0.14) !important;
  border: 1px solid rgba(255,255,255,0.40) !important;
  border-radius: 14px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *{ color: #ffffff !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button{
  background: rgba(255,255,255,0.18) !important;
  border: 1px solid rgba(255,255,255,0.28) !important;
  color: #ffffff !important;
  border-radius: 12px !important;
}

.hero {
  background: #f8fafc;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 18px 20px;
}
.hero-title {
  font-size: 1.4rem;
  font-weight: 800;
  color: #000000;
  margin: 0;
}
.hero-sub {
  font-size: 0.95rem;
  color: #000000;
  opacity: 0.75;
  margin-top: 4px;
}
.card {
  background: #ffffff;
  border-radius: 16px;
  padding: 16px;
  border: 1px solid #e5e7eb;
  box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}
.card h4 {
  font-size: 0.9rem;
  margin-bottom: 6px;
  color: #000000;
  opacity: 0.70;
}
.card .big {
  font-size: 1.6rem;
  font-weight: 800;
  color: #000000;
  margin: 0;
}
.badge {
  display: inline-block;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 0.8rem;
  background: #eff6ff;
  color: #000000;
  border: 1px solid #bfdbfe;
  font-weight: 700;
}
[data-testid="stDataFrame"] {
  border-radius: 14px;
  border: 1px solid #e5e7eb;
  overflow: hidden;
}
.stButton button {
  background: #1d4ed8 !important;
  color: white !important;
  border-radius: 12px !important;
  padding: 0.6rem 1rem !important;
  font-weight: 700 !important;
  border: none !important;
}
.stButton button:hover {
  background: #1e40af !important;
  color: white !important;
}
[data-testid="stMetric"] {
  background: #ffffff;
  border-radius: 14px;
  padding: 12px;
  border: 1px solid #e5e7eb;
}
[data-testid="stMetric"] * { color: #000000 !important; }

button[data-baseweb="tab"] {
  background: transparent;
  font-weight: 700;
  color: #000000;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: #000000;
  border-bottom: 3px solid #1d4ed8;
}
details {
  background: #f9fafb;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  padding: 8px;
}
details * { color: #000000 !important; }
.stCaption, .stCaption * { color: #000000 !important; opacity: 0.75; }

.sidebar-logo {
  position: fixed;
  bottom: 18px;
  left: 18px;
  width: 180px;
}
.sidebar-logo p {
  margin: 6px 0 0 0;
  font-size: 12px;
  color: rgba(255,255,255,0.88) !important;
}
.vega-embed, .vega-embed * { background: #ffffff !important; }
.vega-embed canvas, .vega-embed svg { background: #ffffff !important; }

/* ===== Radio -> Button Tabs (Sidebar) [ONLY OPTIONS] ===== */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"]{
  display: flex !important;
  flex-direction: row !important;
  gap: 10px !important;
  width: 100% !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label{
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
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child{
  display: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label *{
  color: rgba(255,255,255,0.92) !important;
  font-weight: 800 !important;
  letter-spacing: 0.2px !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover{
  background: rgba(255,255,255,0.22) !important;
  border-color: rgba(255,255,255,0.55) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 10px 22px rgba(0,0,0,0.16) !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:active{
  transform: translateY(0px) scale(0.99) !important;
  box-shadow: 0 6px 14px rgba(0,0,0,0.12) !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked){
  background: #ffffff !important;
  border-color: #ffffff !important;
  box-shadow: 0 12px 26px rgba(0,0,0,0.20) !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) *{
  color: #002D72 !important;
  font-weight: 900 !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:focus-visible){
  outline: 3px solid rgba(255,255,255,0.65) !important;
  outline-offset: 2px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
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
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format file tidak didukung. Upload CSV atau XLSX.")
    return df

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
                ("color", "black"),
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

def ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL not in df.columns:
        st.error(f"Kolom tanggal `{DATE_COL}` tidak ditemukan. Pastikan ada kolom call_date.")
        st.stop()
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out = out.dropna(subset=[DATE_COL])
    if out.empty:
        st.warning("Semua call_date gagal diparse. Pastikan format tanggal valid (YYYY-MM-DD).")
        st.stop()
    return out

def parse_metadata_datecall(series: pd.Series) -> pd.Series:
    """
    Parse format seperti: '01 November 2025 08:23:18' (bulan Indonesia).
    Output: datetime64[ns] atau NaT jika gagal.
    """
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

def run_performance_block(df_base: pd.DataFrame, header_badges_html: str, title_context: str):
    df_base = filter_call_types(df_base)
    if df_base.empty:
        st.warning("Tidak ada data setelah filter call type.")
        st.stop()

    df_base = ensure_date(df_base)

    df_base["_month"] = df_base[DATE_COL].dt.to_period("M").astype(str)
    month_list = sorted(df_base["_month"].unique().tolist())
    with st.sidebar:
        selected_month = st.selectbox("Bulan dan Tahun yang terdeteksi: ", month_list, index=len(month_list) - 1)

    dfm = df_base[df_base["_month"] == selected_month].copy()
    if dfm.empty:
        st.warning("Tidak ada data untuk bulan terpilih.")
        st.stop()

    aspect_cols = [c for c in ASPECT_COLUMNS_CANDIDATES if c in dfm.columns]
    missing_aspects = [c for c in ASPECT_COLUMNS_CANDIDATES if c not in dfm.columns]
    if not aspect_cols:
        st.error("Tidak ada kolom aspek yang ditemukan di file untuk dihitung.")
        st.stop()

    yy, mm = selected_month.split("-")
    yy, mm = int(yy), int(mm)
    week_ranges = week_ranges_sun_sat_for_month(yy, mm)

    rows = []
    for col in aspect_cols:
        row = {"Aspek": ASPECT_FRIENDLY_NAMES.get(col, col)}
        for i, (ws, we) in enumerate(week_ranges, start=1):
            mask = (dfm[DATE_COL] >= ws) & (dfm[DATE_COL] <= we)
            df_w = dfm.loc[mask]
            vals_w = df_w[col].apply(normalize_to_binary)
            pct_w = safe_pct(vals_w)
            row[f"Minggu {i} (%)"] = pct_w if not np.isnan(pct_w) else np.nan
        vals_m = dfm[col].apply(normalize_to_binary)
        pct_m = safe_pct(vals_m)
        row["Persentase Bulanan (%)"] = pct_m if not np.isnan(pct_m) else np.nan
        row["Grade"] = tier_from_percent(pct_m) if not np.isnan(pct_m) else "-"
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values(by="Persentase Bulanan (%)", ascending=True)
    overall_monthly = round(float(pd.to_numeric(result_df["Persentase Bulanan (%)"], errors="coerce").mean()), 2)

    left, right = st.columns([1.4, 1.0], vertical_alignment="center")
    with left:
        st.markdown(
            f"""<div class="card">
<h4>{title_context}</h4>
{header_badges_html}
</div>""",
            unsafe_allow_html=True
        )
    with right:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="card"><h4>Overall Bulanan</h4><p class="big">{overall_monthly:.2f}%</p></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="card"><h4>Jumlah Rekaman</h4><p class="big">{len(dfm)}</p></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="card"><h4>Aspek Dihitung</h4><p class="big">{len(aspect_cols)}</p></div>', unsafe_allow_html=True)

    st.write("")
    if missing_aspects:
        with st.expander("ℹ️ Beberapa kolom aspek tidak ditemukan (aman, hanya di-skip)"):
            st.write(missing_aspects)

    tab1, tab2, tab_hour, tab3 = st.tabs(["📊 Overview", "📈 Weekly Trend", "⏰ Hourly Trend", "🧾 Data & Detail"])

    with tab1:
        st.subheader("Ringkasan Performa Bulanan per Aspek")
        show_df = result_df.copy()
        show_df["Grade"] = show_df["Grade"].apply(grade_badge)
        pct_cols = [c for c in show_df.columns if "(%)" in c]
        show_df_display = show_df.copy()
        for c in pct_cols:
            show_df_display[c] = show_df_display[c].apply(lambda x: "Tidak ada rekaman" if pd.isna(x) else f"{x:.2f}%")
        st.dataframe(light_table(show_df_display), use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Trend Overall per Minggu")
        weekly = []
        for i, (ws, we) in enumerate(week_ranges, start=1):
            vals = []
            df_w = dfm[(dfm[DATE_COL] >= ws) & (dfm[DATE_COL] <= we)]
            for c in aspect_cols:
                if not df_w.empty:
                    vals.append(safe_pct(df_w[c].apply(normalize_to_binary)))
            weekly.append({"Minggu": f"M{i}", "Overall": round(float(np.nanmean(vals)), 2)})
        chart_df = pd.DataFrame(weekly)
        chart = alt.Chart(chart_df).mark_line(point=True).encode(
            x=alt.X("Minggu:N", title="Minggu"),
            y=alt.Y("Overall:Q", title="Overall (%)")
        ).properties(height=340, background="white")
        st.altair_chart(chart, use_container_width=True)

    with tab_hour:
        st.subheader("Performa Overall per Jam (08:00–17:00)")
        st.caption("Overall per jam = rata-rata semua aspek pada jam tersebut. Dihitung dari metadata_dateCall.")

        if DATETIME_COL not in dfm.columns:
            st.warning(f"Kolom `{DATETIME_COL}` tidak ditemukan. Hourly Trend tidak bisa dihitung.")
        else:
            dfh = dfm.copy()
            dfh["_dt_call"] = parse_metadata_datecall(dfh[DATETIME_COL])
            dfh = dfh.dropna(subset=["_dt_call"])

            if dfh.empty:
                st.warning("Semua metadata_dateCall gagal diparse. Pastikan format seperti: 01 November 2025 08:23:18")
            else:
                dfh["_hour"] = dfh["_dt_call"].dt.hour
                hours = list(range(8, 18))
                hourly_rows = []

                for h in hours:
                    d = dfh[dfh["_hour"] == h]
                    if d.empty:
                        hourly_rows.append({"Jam": f"{h:02d}:00", "Overall": np.nan, "Jumlah Rekaman": 0, "Hour": h})
                        continue
                    vals = [safe_pct(d[c].apply(normalize_to_binary)) for c in aspect_cols]
                    overall_h = round(float(np.nanmean(vals)), 2)
                    hourly_rows.append({"Jam": f"{h:02d}:00", "Overall": overall_h, "Jumlah Rekaman": len(d), "Hour": h})

                hour_df = pd.DataFrame(hourly_rows).sort_values("Hour")
                df_kpi = hour_df.dropna(subset=["Overall"]).copy()
                total_calls_workhour = int(hour_df["Jumlah Rekaman"].sum())
                best = df_kpi.loc[df_kpi["Overall"].idxmax()] if not df_kpi.empty else None
                worst = df_kpi.loc[df_kpi["Overall"].idxmin()] if not df_kpi.empty else None
                avg_overall = float(df_kpi["Overall"].mean()) if not df_kpi.empty else np.nan

                k1, k2, k3 = st.columns(3)
                if best is not None:
                    k1.markdown(f'<div class="card"><h4>Best Hour</h4><p class="big">{best["Jam"]} • {best["Overall"]:.2f}%</p></div>', unsafe_allow_html=True)
                    k2.markdown(f'<div class="card"><h4>Worst Hour</h4><p class="big">{worst["Jam"]} • {worst["Overall"]:.2f}%</p></div>', unsafe_allow_html=True)
                else:
                    k1.markdown('<div class="card"><h4>Best Hour</h4><p class="big">-</p></div>', unsafe_allow_html=True)
                    k2.markdown('<div class="card"><h4>Worst Hour</h4><p class="big">-</p></div>', unsafe_allow_html=True)
                k3.markdown(f'<div class="card"><h4>Total Rekaman (08–17)</h4><p class="big">{total_calls_workhour}</p></div>', unsafe_allow_html=True)
                st.write("")

                base = alt.Chart(hour_df).encode(
                    x=alt.X("Jam:N", sort=[f"{h:02d}:00" for h in hours], title="Jam", axis=alt.Axis(labelAngle=0))
                )

                area = base.mark_area(opacity=0.18).encode(y=alt.Y("Overall:Q", title="Overall (%)"))
                line = base.mark_line(point=alt.OverlayMarkDef(size=80), strokeWidth=4).encode(
                    y=alt.Y("Overall:Q", title="Overall (%)"),
                    tooltip=[
                        alt.Tooltip("Jam:N", title="Jam"),
                        alt.Tooltip("Overall:Q", title="Overall (%)", format=".2f"),
                        alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                    ],
                )

                if not np.isnan(avg_overall):
                    rule = alt.Chart(pd.DataFrame({"avg": [avg_overall]})).mark_rule(strokeDash=[6, 6]).encode(y="avg:Q")
                    label = alt.Chart(pd.DataFrame({"avg": [avg_overall], "txt": [f"Avg {avg_overall:.1f}%"]})).mark_text(
                        align="left", dx=6, dy=-6
                    ).encode(y="avg:Q", text="txt:N")
                    top_chart = area + line + rule + label
                else:
                    top_chart = area + line

                top_chart = top_chart.properties(height=320, background="white").configure_view(stroke=None, fill="white").configure_axis(
                    labelColor="black", titleColor="black", gridColor="#e5e7eb"
                )
                st.altair_chart(top_chart, use_container_width=True)

                st.write("")
                st.caption("Volume Rekaman per Jam")
                bar = alt.Chart(hour_df).mark_bar().encode(
                    x=alt.X("Jam:N", sort=[f"{h:02d}:00" for h in hours], title="Jam"),
                    y=alt.Y("Jumlah Rekaman:Q", title="Jumlah Rekaman"),
                    tooltip=[alt.Tooltip("Jam:N", title="Jam"), alt.Tooltip("Jumlah Rekaman:Q", title="Jumlah Rekaman")],
                ).properties(height=160, background="white").configure_view(stroke=None, fill="white").configure_axis(
                    labelColor="black", titleColor="black", gridColor="#e5e7eb"
                )
                st.altair_chart(bar, use_container_width=True)

                with st.expander("Detail angka per jam"):
                    st.dataframe(light_table(hour_df[["Jam", "Overall", "Jumlah Rekaman"]]), use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Detail Data untuk Scoring")
        st.dataframe(light_table(dfm.head(50)), use_container_width=True)

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
        Monitoring performa Mitra (Agent) dan Team Leader (TL) per minggu, per jam (08–17), dan ringkasan bulanan per aspek.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

if uploaded is None:
    st.info("Upload file QC dulu di sidebar untuk mulai.")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Gagal baca file: {e}")
    st.stop()

missing = [c for c in [TL_COL, AGENT_COL] if c not in df.columns]
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {missing}\n\nKolom yang ada: {list(df.columns)}")
    st.stop()

df_clean = normalize_identity_cols(df)

with st.sidebar:
    st.markdown("### Kategori Penilaian")
    mode = st.radio("Pilih yang dinilai:", ["Agent", "TL"], horizontal=True, key="mode_penilaian")

tl_list = sorted(df_clean[TL_COL].unique().tolist())

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
    run_performance_block(df_base=df_sel, header_badges_html=header_badges, title_context="Mitra yang terpilih:")

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
    run_performance_block(df_base=df_sel, header_badges_html=header_badges, title_context="Team Leader yang dipilih:")

st.write("")
st.caption("© QC Audio Dashboard")

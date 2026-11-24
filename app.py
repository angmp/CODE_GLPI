"""KPN Corp | Software Inventory Classification.

Streamlit application for offline, human-reviewed GLPI software recommendations.
Compatible with the GridSearchCV Pipeline models created in the thesis notebook,
and with the older estimator + separate TF-IDF artifact format.

This application NEVER writes to GLPI or sends inventory to a remote service.
Only load .joblib artifacts you created and trust.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET
import html
import re

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
HASIL = BASE / "Hasil"
LOGO = BASE / "kpn_logo.png"
MODEL_FILES = {
    "Logistic Regression": "Model_Logistic_Regression.joblib",
    "Linear SVM": "Model_Linear_SVM.joblib",
    "Multinomial Naive Bayes": "Model_Multinomial_Naive_Bayes.joblib",
}
TFIDF_FILE = "TFIDF_Akhir.joblib"  # Only used for legacy, non-Pipeline models
PRED_COLS = ["Prediksi_LR", "Prediksi_SVM", "Prediksi_NB"]
STATUSES = ["INCLUDE", "EXCLUDE", "REVIEW"]
REVIEW_OPTIONS = ["Belum ditinjau", "Setujui INCLUDE", "Setujui EXCLUDE", "Perlu diskusi"]
PALETTE = {"INCLUDE": "#287E65", "EXCLUDE": "#A13A47", "REVIEW": "#BB8542"}
ACCENT = "#8B2935"

st.set_page_config(
    page_title="Software Inventory | KPN Corp",
    page_icon=str(LOGO) if LOGO.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* KPN Design System: quiet backgrounds, crisp type, one burgundy accent. */
:root { --accent:#8B2935; --ink:#29292D; --muted:#77777E; --line:#E8E5E5; }
html, body, [class*="css"] { font-family: Inter, "Segoe UI", system-ui, sans-serif; }
[data-testid="stAppViewContainer"] { background:#F8F8F7; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #EAE8E7; }
[data-testid="stSidebar"] [data-testid="stImage"] img { max-width:220px; }
[data-testid="stSidebar"] .stRadio [role="radiogroup"] { gap:3px; }
[data-testid="stSidebar"] .stRadio label { padding:8px 10px; border-radius:9px; transition:background .15s ease; }
[data-testid="stSidebar"] .stRadio label:hover { background:#F7F1F2; }
[data-testid="stSidebar"] .stRadio label:has(input:checked) { background:#F7EEEE; color:#7D2732; font-weight:650; }
.block-container { max-width:1330px; padding-top:2.15rem; padding-bottom:3rem; }
h1, h2, h3 { color:#242429; letter-spacing:-.035em; }
h1 { font-size:2rem !important; font-weight:750 !important; }
h2, h3 { font-size:1.22rem !important; font-weight:700 !important; }
[data-testid="stDataFrame"] { border:1px solid #E8E5E5; border-radius:12px; overflow:hidden; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:#E8E5E5 !important; }
.stButton>button, .stDownloadButton>button { border-radius:9px; font-weight:590; border-color:#D8D2D2; }
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] { background:#8B2935; color:#FFF; border:1px solid #8B2935; box-shadow:none; }
.stButton>button[kind="primary"]:hover, .stDownloadButton>button[kind="primary"]:hover { background:#70232C; border-color:#70232C; }
[data-testid="stFileUploaderDropzone"] { border-radius:11px; background:#FFF; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { border-radius:9px; }
[data-testid="stProgress"] > div > div { background:#8B2935; }
hr { border-color:#E9E7E6; }
.kpn-eyebrow { color:#8B2935; font-size:11px; letter-spacing:.14em; font-weight:800; text-transform:uppercase; margin-bottom:5px; }
.kpn-pagehead { margin-bottom:24px; }
.kpn-h1 { font-weight:760; color:#27272C; letter-spacing:-.045em; font-size:32px; line-height:1.2; margin:0 0 8px; }
.kpn-sub { font-size:13px; color:#77767C; line-height:1.65; max-width:820px; }
.kpn-brandline { color:#9B8A8B; font-size:11px; letter-spacing:.11em; text-transform:uppercase; font-weight:700; margin-top:7px; }
.kpn-side-rule { height:1px; background:#EDE9E8; margin:22px 0 20px; }
.kpn-side-title { color:#29272B; font-size:13px; font-weight:720; letter-spacing:-.01em; margin-top:8px; }
.kpn-side-desc { color:#88848A; font-size:11px; line-height:1.6; margin-top:6px; }
.kpn-side-status { background:#F7F4F3; border:1px solid #ECE8E6; border-radius:10px; padding:11px 12px; margin:16px 0 6px; }
.kpn-status-name { color:#736B6E; font-size:10px; font-weight:750; letter-spacing:.07em; text-transform:uppercase; }
.kpn-status-value { color:#373237; font-size:12px; margin-top:5px; line-height:1.45; font-weight:620; }
.kpn-kpi { background:#FFF; border:1px solid #E8E5E5; border-radius:13px; min-height:117px; padding:18px 19px; }
.kpn-kpi-label { color:#79767A; font-size:12px; font-weight:650; }
.kpn-kpi-value { color:#29272C; font-weight:780; font-size:30px; line-height:1.25; margin:9px 0 3px; letter-spacing:-.045em; }
.kpn-kpi-note { color:#999397; font-size:11px; }
.kpn-kpi-rule { width:28px; height:3px; border-radius:5px; margin:0 0 12px; }
.kpn-section { margin-top:23px; margin-bottom:12px; }
.kpn-section-title { font-weight:720; font-size:17px; letter-spacing:-.025em; color:#29272C; }
.kpn-section-sub { margin-top:5px; font-size:12px; color:#848187; }
.kpn-card { background:#FFF; border:1px solid #E8E5E5; border-radius:13px; padding:20px 21px; }
.kpn-card-eyebrow { font-size:10px; font-weight:750; letter-spacing:.10em; color:#8B2935; text-transform:uppercase; }
.kpn-card-num { font-size:23px; font-weight:750; color:#2D2B2E; margin:7px 0 4px; }
.kpn-card-caption { color:#88848A; font-size:12px; }
.kpn-note { background:#FFF; border:1px solid #E8E5E5; border-left:3px solid #8B2935; border-radius:10px; padding:13px 17px; color:#66636A; font-size:12px; line-height:1.65; }
.kpn-footer { border-top:1px solid #E8E5E5; color:#999399; font-size:11px; padding-top:14px; margin-top:34px; }
@media (max-width: 850px) { .block-container { padding-top:1rem; } .kpn-h1 { font-size:26px; } }
</style>
""",
    unsafe_allow_html=True,
)


def fmt(n: int | float) -> str:
    return f"{int(n):,}".replace(",", ".")


def page_head(section: str, title: str, subtitle: str) -> None:
    st.markdown(
        '<div class="kpn-pagehead"><div class="kpn-eyebrow">'
        + html.escape(section)
        + '</div><div class="kpn-h1">'
        + html.escape(title)
        + '</div><div class="kpn-sub">'
        + html.escape(subtitle)
        + '</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    st.markdown(
        '<div class="kpn-section"><div class="kpn-section-title">'
        + html.escape(title)
        + '</div><div class="kpn-section-sub">'
        + html.escape(subtitle)
        + '</div></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str, line_color: str = ACCENT) -> None:
    st.markdown(
        '<div class="kpn-kpi"><div class="kpn-kpi-rule" style="background:'
        + line_color
        + '"></div><div class="kpn-kpi-label">'
        + html.escape(label)
        + '</div><div class="kpn-kpi-value">'
        + html.escape(value)
        + '</div><div class="kpn-kpi-note">'
        + html.escape(note)
        + '</div></div>',
        unsafe_allow_html=True,
    )


def normalize(series: pd.Series) -> pd.Series:
    """Same preprocessing as the notebook: lowercase, trim, collapse whitespace."""
    return series.fillna("").astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)


def model_signatures() -> tuple:
    return tuple(
        (filename, (HASIL / filename).stat().st_mtime_ns if (HASIL / filename).exists() else None)
        for filename in [*MODEL_FILES.values(), TFIDF_FILE]
    )


def missing_model_files() -> list[str]:
    return [name for name in MODEL_FILES.values() if not (HASIL / name).is_file()]


@st.cache_resource(show_spinner="Menyiapkan model klasifikasi...")
def load_models(signatures: tuple) -> tuple[dict, object | None]:
    """Load only locally trusted joblib. Pipeline is the preferred final format."""
    models = {key: joblib.load(HASIL / filename) for key, filename in MODEL_FILES.items()}
    legacy = any(not isinstance(model, Pipeline) for model in models.values())
    tfidf = None
    if legacy:
        path = HASIL / TFIDF_FILE
        if not path.is_file():
            raise FileNotFoundError(
                "Model lama terdeteksi, tetapi Hasil/TFIDF_Akhir.joblib tidak tersedia. "
                "Gunakan Pipeline dari GridSearchCV atau tambahkan TF-IDF yang sesuai."
            )
        tfidf = joblib.load(path)
    return models, tfidf


def get_models() -> tuple[dict, object | None]:
    missing = missing_model_files()
    if missing:
        raise FileNotFoundError("File model belum tersedia: " + ", ".join(missing))
    return load_models(model_signatures())


def parse_csv(raw: bytes, delimiter: str = "Otomatis", require_name: bool = True) -> tuple[pd.DataFrame, str, str]:
    separators = {"Koma": [","], "Titik koma": [";"], "Tab": ["\t"], "Pipe": ["|"]}
    choices = [",", ";", "\t", "|"] if delimiter == "Otomatis" else separators[delimiter]
    candidates: list[tuple[int, pd.DataFrame, str, str]] = []
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        for sep in choices:
            try:
                data = pd.read_csv(
                    BytesIO(raw), encoding=encoding, sep=sep, dtype=str, keep_default_na=False,
                    on_bad_lines="error",
                )
            except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue
            data.columns = data.columns.astype(str).str.strip().str.lstrip("\ufeff")
            if "name" in [c.lower() for c in data.columns]:
                found = next(c for c in data.columns if c.lower() == "name")
                if found != "name":
                    data = data.rename(columns={found: "name"})
                return data, encoding, repr(sep)
            if len(data.columns) > 1:
                candidates.append((len(data.columns), data, encoding, repr(sep)))
    if not require_name and candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, data, encoding, sep = candidates[0]
        return data, encoding, sep
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, data, encoding, sep = candidates[0]
        return data, encoding, sep  # Caller may select a different name column.
    raise ValueError("CSV tidak dapat dibaca. Periksa encoding dan pemisah kolom.")


def predict(frame: pd.DataFrame) -> pd.DataFrame:
    if "name" not in frame:
        raise ValueError("Kolom 'name' tidak ditemukan pada dataset.")
    models, tfidf = get_models()
    result = frame.copy().reset_index(drop=True)
    cleaned = normalize(result["name"])
    valid = cleaned.ne("")
    result["nama_bersih"] = cleaned
    for col in PRED_COLS:
        result[col] = ""
    result["Status_Prediksi"] = "REVIEW"
    result["Alasan_Status"] = "Nama software kosong"
    if valid.any():
        texts = cleaned.loc[valid]
        transformed = None
        predictions: dict[str, np.ndarray] = {}
        for (model_name, model), column in zip(models.items(), PRED_COLS):
            if isinstance(model, Pipeline):
                pred = model.predict(texts)
            else:
                if transformed is None:
                    transformed = tfidf.transform(texts)
                pred = model.predict(transformed)
            predictions[column] = np.asarray(pred, dtype=str)
            result.loc[valid, column] = predictions[column]
        p1, p2, p3 = (predictions[col] for col in PRED_COLS)
        agreed = (p1 == p2) & (p2 == p3)
        result.loc[valid, "Status_Prediksi"] = np.where(agreed, p1, "REVIEW")
        result.loc[valid, "Alasan_Status"] = np.where(
            agreed, "Ketiga model sepakat", "Prediksi model berbeda"
        )
    result["Keputusan_IT"] = "Belum ditinjau"
    result["Catatan_IT"] = ""
    result["_rowid"] = np.arange(len(result))
    return result


def choose_inventory_file() -> Path | None:
    for candidate in [HASIL / "Data_Kolom_Digunakan.csv", BASE / "RawData.csv"]:
        if candidate.is_file():
            return candidate
    return None


def load_default_inventory() -> tuple[pd.DataFrame | None, str]:
    path = choose_inventory_file()
    if path is None:
        return None, "Dataset belum ditemukan"
    data, _, _ = parse_csv(path.read_bytes())
    if "name" not in data:
        return None, "Dataset tidak memiliki kolom name"
    return predict(data), path.name


def distribution(frame: pd.DataFrame) -> pd.DataFrame:
    count = frame["Status_Prediksi"].value_counts()
    return pd.DataFrame({"Status": STATUSES, "Jumlah": [int(count.get(s, 0)) for s in STATUSES]})


def display_chart(frame: pd.DataFrame) -> None:
    data = distribution(frame)
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=5, size=22).encode(
        y=alt.Y("Status:N", sort=STATUSES, title=None, axis=alt.Axis(labelFontSize=12, labelColor="#605D63")),
        x=alt.X("Jumlah:Q", title=None, axis=alt.Axis(gridColor="#EDE9E8", domain=False, tickCount=5)),
        color=alt.Color("Status:N", scale=alt.Scale(domain=STATUSES, range=[PALETTE[x] for x in STATUSES]), legend=None),
        tooltip=["Status:N", alt.Tooltip("Jumlah:Q", format=",.0f")],
    ).properties(height=220)
    st.altair_chart(bars, use_container_width=True)


def data_table(frame: pd.DataFrame, rows: int | None = None) -> None:
    cols = [col for col in [
        "name", "manufacturer_name", "Prediksi_LR", "Prediksi_SVM", "Prediksi_NB",
        "Status_Prediksi", "Keputusan_IT", "Alasan_Status"
    ] if col in frame.columns]
    st.dataframe(
        frame[cols].head(rows) if rows else frame[cols],
        hide_index=True, use_container_width=True, height=360,
        column_config={"name": st.column_config.TextColumn("Nama software", width="large"),
                       "Status_Prediksi": st.column_config.TextColumn("Rekomendasi")},
    )


def safe_export(frame: pd.DataFrame) -> pd.DataFrame:
    """Escape potential spreadsheet formulas from untrusted GLPI software names."""
    output = frame.drop(columns=["_rowid"], errors="ignore").copy()
    for col in output.select_dtypes(include=["object", "string"]).columns:
        output[col] = output[col].map(
            lambda x: "'" + x if isinstance(x, str) and re.match(r"^[\s\t\r]*[=+@-]", x) else x
        )
    return output


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return safe_export(frame).to_csv(index=False).encode("utf-8-sig")


def excel_bytes(frame: pd.DataFrame) -> bytes:
    stream = BytesIO()
    output = safe_export(frame)
    with pd.ExcelWriter(stream, engine="openpyxl") as writer:
        output.to_excel(writer, sheet_name="Semua", index=False)
        for status in STATUSES:
            output.loc[output["Status_Prediksi"] == status].to_excel(
                writer, sheet_name=status, index=False
            )
        distribution(output).to_excel(writer, sheet_name="Ringkasan", index=False)
    return stream.getvalue()


def review_xml(frame: pd.DataFrame) -> bytes:
    """Audit-list only, intentionally NOT the GLPI rules import format."""
    root = ET.Element("review-list", {"glpi-import-compatible": "false"})
    ET.SubElement(root, "notice").text = "Daftar audit, bukan XML impor aturan GLPI."
    for _, row in frame.iterrows():
        element = ET.SubElement(root, "software")
        for tag, key in [("name", "name"), ("status", "Status_Prediksi"),
                         ("decision", "Keputusan_IT"), ("note", "Catatan_IT")]:
            ET.SubElement(element, tag).text = str(row.get(key, ""))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def metric_data() -> pd.DataFrame | None:
    path = HASIL / "Tabel_4_8_Evaluasi_Tiga_Model.csv"
    if not path.exists():
        return None
    df, _, _ = parse_csv(path.read_bytes(), require_name=False)
    if "Model" not in df:
        return None
    for col in ["Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "Recall_INCLUDE"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ----- Sidebar / persistent app state -----
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=220)
    else:
        st.markdown("### KPN CORP")
    st.markdown('<div class="kpn-brandline">IT ASSET MANAGEMENT</div>', unsafe_allow_html=True)
    st.markdown('<div class="kpn-side-rule"></div>', unsafe_allow_html=True)
    navigation = st.radio(
        "Navigasi", ["Ringkasan", "Klasifikasi", "Inventaris", "Evaluasi Model", "Ekspor Data"],
        label_visibility="collapsed"
    )
    missing = missing_model_files()
    status_message = "Tiga model tersedia" if not missing else f"{len(missing)} model belum tersedia"
    st.markdown(
        '<div class="kpn-side-status"><div class="kpn-status-name">Status model</div>'
        '<div class="kpn-status-value">' + html.escape(status_message) + '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="kpn-side-desc">Pemrosesan lokal. Prediksi hanya rekomendasi dan tidak mengubah data atau aturan GLPI.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Muat ulang data dan model", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.pop("hasil", None)
        st.session_state.pop("hasil_input", None)
        st.session_state.pop("hasil_source", None)
        st.rerun()

if "hasil" not in st.session_state and not missing:
    try:
        loaded, src = load_default_inventory()
        if loaded is not None:
            st.session_state["hasil"] = loaded
            st.session_state["hasil_source"] = src
    except Exception as exc:
        st.session_state["initial_error"] = str(exc)

result: pd.DataFrame | None = st.session_state.get("hasil")

if navigation == "Ringkasan":
    page_head("KPN CORP / SOFTWARE INVENTORY", "Ringkasan klasifikasi", "Gambaran rekomendasi pemilahan software berdasarkan hasil prediksi tiga algoritma machine learning.")
    if result is None:
        st.info("Data inventaris belum dimuat. Simpan file Hasil/Data_Kolom_Digunakan.csv dan tiga model, atau gunakan menu Klasifikasi untuk mengunggah CSV.")
        if st.session_state.get("initial_error"):
            st.caption("Detail pemuatan data: " + st.session_state["initial_error"])
    else:
        totals = dict(zip(distribution(result)["Status"], distribution(result)["Jumlah"]))
        cards = st.columns(4, gap="medium")
        with cards[0]: metric_card("Total entri", fmt(len(result)), "Software Inventory GLPI", ACCENT)
        with cards[1]: metric_card("INCLUDE", fmt(totals["INCLUDE"]), "Rekomendasi ditampilkan", PALETTE["INCLUDE"])
        with cards[2]: metric_card("EXCLUDE", fmt(totals["EXCLUDE"]), "Rekomendasi tidak ditampilkan", PALETTE["EXCLUDE"])
        with cards[3]: metric_card("REVIEW", fmt(totals["REVIEW"]), "Perlu keputusan tim IT", PALETTE["REVIEW"])
        left, right = st.columns([1.6, 1], gap="large")
        with left:
            section("Distribusi rekomendasi", "Jumlah entri berdasarkan kesepakatan ketiga model")
            display_chart(result)
        with right:
            section("Kesepakatan model", "Persentase tiga model memberikan kelas yang sama")
            agreement = (len(result) - totals["REVIEW"]) / len(result) if len(result) else 0.0
            st.markdown(
                f'<div class="kpn-card"><div class="kpn-card-eyebrow">Konsensus prediksi</div>'
                f'<div class="kpn-card-num">{agreement:.1%}</div>'
                '<div class="kpn-card-caption">Bukan metrik akurasi; hanya kesamaan keluaran antarmodel.</div></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.progress(float(agreement))
            st.caption("Sumber: " + st.session_state.get("hasil_source", "Data sesi aktif"))
        section("Prioritas peninjauan", "Contoh perangkat lunak yang perlu pemeriksaan lebih lanjut")
        pending = result.loc[result["Status_Prediksi"] == "REVIEW"]
        if len(pending):
            data_table(pending, rows=10)
        else:
            st.info("Tidak ada entri REVIEW pada data saat ini.")

elif navigation == "Klasifikasi":
    page_head("PREDIKSI / TIGA MODEL", "Klasifikasi perangkat lunak", "Masukkan nama software atau unggah dataset. Setiap nama diprediksi oleh ketiga algoritma dengan model yang tersimpan secara lokal.")
    if missing:
        st.error("Model belum lengkap pada folder Hasil/: " + ", ".join(missing))
    tab_manual, tab_csv = st.tabs(["Input nama software", "Unggah data CSV"])
    with tab_manual:
        with st.form("predict_manual", clear_on_submit=False):
            names = st.text_area(
                "Nama software (satu nama per baris)", height=155,
                placeholder="Microsoft Office 2021\nSketchUp Pro 2024\nNVIDIA Graphics Driver",
            )
            run = st.form_submit_button("Jalankan klasifikasi", type="primary", disabled=bool(missing))
        if run:
            lines = [line.strip() for line in names.splitlines() if line.strip()]
            if not lines:
                st.warning("Masukkan minimal satu nama software.")
            else:
                try:
                    st.session_state["hasil_input"] = predict(pd.DataFrame({"name": lines}))
                    st.session_state["input_source"] = "Input manual"
                except Exception as exc:
                    st.error("Prediksi tidak dapat dijalankan: " + str(exc))
    with tab_csv:
        file = st.file_uploader("Pilih file CSV Software Inventory", type=["csv"])
        sep = st.selectbox("Pemisah kolom", ["Otomatis", "Koma", "Titik koma", "Tab", "Pipe"])
        csv_frame = None
        column_choice = None
        if file is not None:
            try:
                csv_frame, encoding, delimiter = parse_csv(file.getvalue(), sep)
                st.caption(f"{fmt(len(csv_frame))} entri, {len(csv_frame.columns)} kolom. Encoding: {encoding}; delimiter: {delimiter}.")
                if "name" not in csv_frame:
                    column_choice = st.selectbox("Kolom nama software", list(csv_frame.columns))
                st.dataframe(csv_frame.head(6), hide_index=True, use_container_width=True)
            except Exception as exc:
                st.error("CSV tidak dapat dibaca: " + str(exc))
        if st.button("Klasifikasikan CSV", type="primary", disabled=bool(missing) or csv_frame is None):
            try:
                source = csv_frame.copy()
                if column_choice:
                    source["name"] = source[column_choice]
                output = predict(source)
                st.session_state["hasil"] = output
                st.session_state["hasil_source"] = file.name if file is not None else "Upload CSV"
                st.session_state["hasil_input"] = output
                st.session_state["input_source"] = "Upload CSV"
                st.success(f"{fmt(len(output))} nama software selesai diklasifikasikan.")
            except Exception as exc:
                st.error("Klasifikasi gagal: " + str(exc))
    current = st.session_state.get("hasil_input")
    if current is not None:
        section("Hasil klasifikasi", "Perbandingan keluaran setiap algoritma. REVIEW berarti prediksi berbeda atau nama kosong.")
        data_table(current, rows=200)
        st.download_button("Unduh hasil prediksi (.csv)", csv_bytes(current),
                           "Hasil_Klasifikasi_GLPI.csv", mime="text/csv")

elif navigation == "Inventaris":
    page_head("DATA INVENTARIS / HUMAN REVIEW", "Inventaris dan peninjauan", "Telusuri hasil prediksi setiap model dan dokumentasikan keputusan tim IT tanpa mengubah inventaris asli di GLPI.")
    if result is None:
        st.info("Belum ada inventaris. Gunakan menu Klasifikasi untuk memproses CSV atau tempatkan data di folder Hasil.")
    else:
        all_tab, review_tab = st.tabs(["Semua software", "Peninjauan tim IT"])
        with all_tab:
            a, b, c = st.columns([2.3, 1.5, 1])
            with a: query = st.text_input("Cari software", placeholder="Cari nama atau kata kunci...")
            with b: statuses = st.multiselect("Status", STATUSES, default=STATUSES)
            with c: cap = st.selectbox("Tampilkan baris", [25, 50, 100, 250, 500], index=2)
            view = result.loc[result["Status_Prediksi"].isin(statuses)]
            if query.strip():
                view = view.loc[view["name"].astype(str).str.contains(query.strip(), case=False, regex=False, na=False)]
            st.caption(f"{fmt(len(view))} data cocok dengan filter")
            data_table(view, rows=cap)
            st.download_button("Unduh hasil sesuai filter", csv_bytes(view), "Inventaris_Terfilter.csv", mime="text/csv")
        with review_tab:
            st.markdown('<div class="kpn-note">Model hanya memberikan rekomendasi. Keputusan IT disimpan pada sesi ini; unduh CSV untuk menyimpan hasil peninjauan secara permanen.</div>', unsafe_allow_html=True)
            query_review = st.text_input("Cari nama untuk ditinjau", placeholder="Cari entri yang perlu dikonfirmasi...")
            review_status = st.multiselect("Tampilkan rekomendasi", STATUSES, default=["REVIEW"], key="review_status")
            only_pending = st.checkbox("Hanya keputusan yang belum ditinjau", value=False)
            view = result.loc[result["Status_Prediksi"].isin(review_status)]
            if query_review:
                view = view.loc[view["name"].astype(str).str.contains(query_review, case=False, regex=False, na=False)]
            if only_pending:
                view = view.loc[view["Keputusan_IT"].eq("Belum ditinjau")]
            st.caption(f"{fmt(len(view))} entri tersedia untuk ditinjau")
            visible = view.head(100).copy()
            columns = ["_rowid", "name", *PRED_COLS, "Status_Prediksi", "Keputusan_IT", "Catatan_IT"]
            if visible.empty:
                st.info("Tidak ada data yang cocok dengan pilihan ini.")
            else:
                edited = st.data_editor(
                    visible[columns], hide_index=True, use_container_width=True,
                    num_rows="fixed", height=355,
                    disabled=[c for c in columns if c not in ("Keputusan_IT", "Catatan_IT")],
                    column_config={
                        "_rowid": st.column_config.NumberColumn("ID baris", format="%d"),
                        "name": st.column_config.TextColumn("Nama software", width="large"),
                        "Keputusan_IT": st.column_config.SelectboxColumn("Keputusan IT", options=REVIEW_OPTIONS, required=True),
                        "Catatan_IT": st.column_config.TextColumn("Catatan", max_chars=300),
                    },
                    key="review_editor",
                )
                if st.button("Simpan keputusan", type="primary"):
                    main = st.session_state["hasil"].copy()
                    for _, edited_row in edited.iterrows():
                        index = int(edited_row["_rowid"])
                        matched = main["_rowid"] == index
                        main.loc[matched, "Keputusan_IT"] = edited_row["Keputusan_IT"]
                        main.loc[matched, "Catatan_IT"] = edited_row["Catatan_IT"]
                    st.session_state["hasil"] = main
                    st.success("Keputusan tersimpan pada sesi dashboard.")
                st.download_button("Unduh hasil dan keputusan IT", csv_bytes(st.session_state["hasil"]),
                                   "Hasil_Review_Tim_IT.csv", mime="text/csv")

elif navigation == "Evaluasi Model":
    page_head("MODEL PERFORMANCE / HOLDOUT TEST", "Evaluasi model", "Metrik berasal dari pengujian notebook setelah optimasi GridSearchCV, bukan dari jumlah kesepakatan prediksi pada dashboard.")
    metrics = metric_data()
    if metrics is None:
        st.info("Belum ada file Hasil/Tabel_4_8_Evaluasi_Tiga_Model.csv. Jalankan kode evaluasi setelah optimasi terlebih dahulu.")
    else:
        if "F1_Macro" in metrics and metrics["F1_Macro"].notna().any():
            best = metrics.loc[metrics["F1_Macro"].idxmax()]
            a, b, c = st.columns(3)
            with a: metric_card("Model F1 macro terbaik", str(best["Model"]), "Berdasarkan data testing", ACCENT)
            with b: metric_card("F1 macro terbaik", f"{best['F1_Macro']:.4f}", "Performa keseimbangan dua kelas", PALETTE["INCLUDE"])
            with c: metric_card("Jumlah model", fmt(len(metrics)), "Algoritma yang dibandingkan", PALETTE["REVIEW"])
        section("Hasil metrik klasifikasi")
        st.dataframe(metrics, hide_index=True, use_container_width=True)
        if {"Model", "F1_Macro"}.issubset(metrics.columns):
            section("Perbandingan F1-score macro")
            plot = alt.Chart(metrics).mark_bar(cornerRadiusEnd=5, size=22, color=ACCENT).encode(
                y=alt.Y("Model:N", sort="-x", title=None),
                x=alt.X("F1_Macro:Q", scale=alt.Scale(domain=[0, 1]), title="F1 macro"),
                tooltip=["Model:N", alt.Tooltip("F1_Macro:Q", format=".4f")],
            ).properties(height=205)
            st.altair_chart(plot, use_container_width=True)
        st.caption("Pastikan CSV evaluasi telah diperbarui dari pred_lr_opt, pred_svm_opt, dan pred_nb_opt.")
        comparison = HASIL / "Tabel_Perbandingan_Optimasi.csv"
        if comparison.exists():
            with st.expander("Lihat perbandingan sebelum dan sesudah optimasi"):
                before_after, _, _ = parse_csv(comparison.read_bytes(), require_name=False)
                st.dataframe(before_after, hide_index=True, use_container_width=True)

elif navigation == "Ekspor Data":
    page_head("DATA OUTPUT / EXPORT", "Ekspor hasil klasifikasi", "Unduh laporan rekomendasi dan hasil peninjauan dalam format CSV atau Excel. Tidak ada perubahan yang dikirimkan ke sistem GLPI.")
    if result is None:
        st.info("Belum ada hasil untuk diekspor.")
    else:
        section("Laporan lengkap", "Hasil prediksi ketiga model beserta keputusan tim IT yang telah dicatat")
        first, second = st.columns(2, gap="medium")
        with first:
            st.download_button("Unduh CSV lengkap", csv_bytes(result),
                               "Hasil_Prediksi_Software.csv", mime="text/csv", type="primary", use_container_width=True)
        with second:
            st.download_button("Unduh Excel (5 sheet)", excel_bytes(result),
                               "Rekomendasi_Software_KPN.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        section("Unduh berdasarkan status")
        boxes = st.columns(3)
        for col, status in zip(boxes, STATUSES):
            subset = result.loc[result["Status_Prediksi"] == status]
            with col:
                st.download_button(f"{status} - {fmt(len(subset))} entri", csv_bytes(subset),
                                   f"Daftar_{status}.csv", mime="text/csv", use_container_width=True)
        with st.expander("Daftar audit keputusan IT (XML non-GLPI)"):
            approved = result.loc[
                (result["Status_Prediksi"] == "EXCLUDE") &
                (result["Keputusan_IT"] == "Setujui EXCLUDE")
            ]
            st.caption("XML ini hanya untuk dokumentasi peninjauan. Bukan format impor aturan Software Dictionary GLPI.")
            st.download_button("Unduh XML daftar audit", review_xml(approved), "Daftar_Audit_Review.xml",
                               mime="application/xml", disabled=approved.empty)

st.markdown(
    '<div class="kpn-footer">KPN Corp / IT Asset Management &nbsp;&middot;&nbsp; '
    'Prototipe penelitian: rekomendasi klasifikasi, bukan perubahan aturan GLPI.</div>',
    unsafe_allow_html=True,
)

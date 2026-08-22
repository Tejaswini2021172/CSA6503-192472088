"""
================================================================================
Q38_Summarizer_Translator | app.py
Industry-level Streamlit application:
Summarizer + Translator Application

Pipeline:
Input -> PDF Reader / Text -> Extractive Summary -> Gemini Summary
-> Translation -> Analytics -> Download

Author: Generative AI Lab - Question 38
================================================================================
"""

import time
import io
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from pdf_reader import extract_text_from_pdf, PDFReadError
from extractive import extractive_summary, ExtractiveSummaryError
from gemini_summary import generate_gemini_summary, GeminiSummaryError
from translator import translate_text, get_supported_languages, TranslationError


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Q38 | Summarizer + Translator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# GLOBAL CSS — GLASSMORPHISM + GRADIENT PREMIUM UI
# ==============================================================================
def load_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', 'Poppins', sans-serif;
        }

        /* ---------- APP BACKGROUND ---------- */
        .stApp {
            background: radial-gradient(circle at 15% 20%, #1e1b4b 0%, #0f0c29 35%, #050508 100%);
            background-attachment: fixed;
            color: #f5f5fa;
        }

        /* Animated aurora backdrop */
        .stApp::before {
            content: "";
            position: fixed;
            top: -20%;
            left: -20%;
            width: 140%;
            height: 140%;
            background: radial-gradient(circle at 20% 30%, rgba(124, 58, 237, 0.25), transparent 40%),
                        radial-gradient(circle at 80% 20%, rgba(59, 130, 246, 0.22), transparent 40%),
                        radial-gradient(circle at 50% 80%, rgba(236, 72, 153, 0.18), transparent 45%);
            animation: auroraMove 18s ease-in-out infinite alternate;
            z-index: -1;
            pointer-events: none;
        }

        @keyframes auroraMove {
            0%   { transform: translate(0px, 0px) scale(1); }
            50%  { transform: translate(30px, -20px) scale(1.08); }
            100% { transform: translate(-20px, 20px) scale(1.03); }
        }

        /* ---------- HERO BANNER ---------- */
        .hero-banner {
            position: relative;
            padding: 3rem 2.5rem;
            border-radius: 28px;
            background: linear-gradient(120deg, rgba(124,58,237,0.35), rgba(59,130,246,0.30), rgba(236,72,153,0.28));
            background-size: 200% 200%;
            animation: gradientShift 10s ease infinite;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 8px 32px rgba(0,0,0,0.35);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            margin-bottom: 2rem;
            text-align: center;
            overflow: hidden;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .hero-banner h1 {
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
            background: linear-gradient(90deg, #ffffff, #c4b5fd, #93c5fd, #ffffff);
            background-size: 300% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shine 6s linear infinite;
        }

        @keyframes shine {
            to { background-position: 300% center; }
        }

        .hero-banner p {
            font-size: 1.15rem;
            color: #e5e7eb;
            font-weight: 400;
            max-width: 780px;
            margin: 0 auto;
        }

        .hero-badges {
            margin-top: 1.2rem;
            display: flex;
            justify-content: center;
            gap: 0.6rem;
            flex-wrap: wrap;
        }

        .hero-badge {
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.25);
            font-size: 0.82rem;
            font-weight: 500;
            color: #f3f4f6;
            backdrop-filter: blur(6px);
        }

        /* ---------- GLASS CARD ---------- */
        .glass-card {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 22px;
            padding: 1.6rem 1.8rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.28);
            margin-bottom: 1.4rem;
            transition: all 0.35s ease;
        }

        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 14px 36px rgba(124, 58, 237, 0.28);
            border: 1px solid rgba(196, 181, 253, 0.45);
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
            color: #f5f3ff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* ---------- METRIC CARD ---------- */
        .metric-card {
            background: linear-gradient(145deg, rgba(124,58,237,0.20), rgba(59,130,246,0.14));
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 18px;
            padding: 1.1rem 1rem;
            text-align: center;
            backdrop-filter: blur(12px);
            transition: transform 0.3s ease;
        }
        .metric-card:hover { transform: translateY(-3px) scale(1.02); }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 800;
            background: linear-gradient(90deg, #c4b5fd, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #d1d5db;
            margin-top: 0.2rem;
            font-weight: 500;
            letter-spacing: 0.02em;
        }

        /* ---------- BUTTONS ---------- */
        div.stButton > button, div.stDownloadButton > button {
            background: linear-gradient(90deg, #7c3aed, #3b82f6);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.7rem 1.6rem;
            font-weight: 600;
            font-size: 1rem;
            box-shadow: 0 4px 18px rgba(124, 58, 237, 0.45);
            transition: all 0.3s ease;
            width: 100%;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            transform: translateY(-2px) scale(1.015);
            box-shadow: 0 8px 26px rgba(124, 58, 237, 0.65);
            background: linear-gradient(90deg, #8b5cf6, #60a5fa);
        }
        div.stButton > button:active, div.stDownloadButton > button:active {
            transform: translateY(0px) scale(0.99);
        }

        /* ---------- TABS ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255,255,255,0.04);
            padding: 0.4rem;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.10);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.6rem 1.1rem;
            font-weight: 600;
            color: #d1d5db;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #7c3aed, #3b82f6) !important;
            color: white !important;
        }

        /* ---------- SIDEBAR ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #14102b 0%, #0b0918 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        section[data-testid="stSidebar"] .glass-card {
            background: rgba(255,255,255,0.05);
        }

        /* ---------- TEXT AREAS / INPUTS ---------- */
        textarea, .stTextInput input {
            background: rgba(255,255,255,0.06) !important;
            color: #f5f5fa !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
        }

        /* ---------- FILE UPLOADER ---------- */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.05);
            border: 1.5px dashed rgba(196,181,253,0.5);
            border-radius: 16px;
        }

        /* ---------- FOOTER ---------- */
        .app-footer {
            margin-top: 3rem;
            padding: 1.8rem;
            border-radius: 20px;
            text-align: center;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            color: #9ca3af;
            font-size: 0.9rem;
        }
        .app-footer span {
            background: linear-gradient(90deg, #c4b5fd, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        /* ---------- BADGES / PILLS ---------- */
        .pill {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }
        .pill-success { background: rgba(16,185,129,0.18); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
        .pill-info { background: rgba(59,130,246,0.18); color: #60a5fa; border: 1px solid rgba(96,165,250,0.4); }
        .pill-warning { background: rgba(245,158,11,0.18); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }

        /* Divider */
        hr {
            border-color: rgba(255,255,255,0.1);
        }

        h1, h2, h3, h4, h5 { color: #f5f3ff; }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: #0f0c29; }
        ::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


load_css()


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
def init_session_state():
    defaults = {
        "raw_text": "",
        "extractive_result": "",
        "gemini_result": "",
        "translated_result": "",
        "target_lang_name": "French",
        "stats": {},
        "pipeline_ran": False,
        "input_source": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ==============================================================================
# HERO BANNER
# ==============================================================================
def render_hero():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🧠 AI Summarizer &amp; Translator Studio</h1>
            <p>Transform lengthy documents into concise, intelligent summaries — then translate them
            instantly into any language. Powered by Google Gemini, Sumy LSA, and Neural Translation.</p>
            <div class="hero-badges">
                <span class="hero-badge">⚡ Gemini AI Powered</span>
                <span class="hero-badge">📄 PDF &amp; TXT Support</span>
                <span class="hero-badge">🌍 100+ Languages</span>
                <span class="hero-badge">📊 Live Analytics</span>
                <span class="hero-badge">☁️ Streamlit Cloud Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_hero()


# ==============================================================================
# SIDEBAR
# ==============================================================================
LANGUAGES = get_supported_languages()

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <h2 style="margin-bottom:0;">⚙️ Control Panel</h2>
            <p style="color:#9ca3af; font-size:0.85rem;">Configure your summarization pipeline</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📥 Input Type</div>', unsafe_allow_html=True)
    input_type = st.radio(
        "Choose your input method",
        options=["Paste Text", "Upload TXT", "Upload PDF"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📏 Summary Length</div>', unsafe_allow_html=True)
    summary_length = st.select_slider(
        "Select summary length",
        options=["Short", "Medium", "Long"],
        value="Medium",
        label_visibility="collapsed",
    )
    length_map = {"Short": 3, "Medium": 6, "Long": 10}
    st.caption(f"Extractive sentences: **{length_map[summary_length]}**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌍 Translate To</div>', unsafe_allow_html=True)
    target_lang_name = st.selectbox(
        "Select target language",
        options=sorted(LANGUAGES.keys()),
        index=sorted(LANGUAGES.keys()).index("french") if "french" in LANGUAGES else 0,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔑 Gemini API Key</div>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "Gemini API Key (optional if set in .env)",
        type="password",
        placeholder="Enter your GEMINI_API_KEY",
        label_visibility="collapsed",
    )
    st.caption("Leave blank to use the key configured in your `.env` file.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">ℹ️ Model Information</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="pill pill-info">Gemini 3.6 Flash</div>
        <div class="pill pill-success">Sumy LSA</div>
        <div class="pill pill-warning">Deep Translator</div>
        <br><br>
        <p style="font-size:0.82rem; color:#9ca3af; line-height:1.5;">
        <b>Extractive:</b> Latent Semantic Analysis (LSA) selects the most
        semantically important sentences directly from source text.<br><br>
        <b>Abstractive:</b> Google Gemini 3.6 Flash generates a fluent,
        human-like paraphrased summary.<br><br>
        <b>Translation:</b> Google Translator engine via deep-translator,
        supporting 100+ languages.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center; margin-top:1rem; color:#6b7280; font-size:0.78rem;">
            Q38 · Generative AI Lab<br>Summarizer + Translator App
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# INPUT SECTION
# ==============================================================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📝 Document Input</div>', unsafe_allow_html=True)

input_text = ""
input_error = None

if input_type == "Paste Text":
    input_text = st.text_area(
        "Paste your document text below",
        height=260,
        placeholder="Paste a lengthy article, research paper, report, or any long document here...",
        label_visibility="collapsed",
    )

elif input_type == "Upload TXT":
    uploaded_txt = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_txt is not None:
        try:
            raw_bytes = uploaded_txt.read()
            input_text = raw_bytes.decode("utf-8", errors="ignore")
            st.success(f"✅ Loaded `{uploaded_txt.name}` ({len(raw_bytes)} bytes)")
            with st.expander("Preview extracted text"):
                st.text(input_text[:2000] + ("..." if len(input_text) > 2000 else ""))
        except Exception as e:
            input_error = f"Failed to read TXT file: {e}"

elif input_type == "Upload PDF":
    uploaded_pdf = st.file_uploader("Upload a .pdf file", type=["pdf"])
    if uploaded_pdf is not None:
        try:
            input_text = extract_text_from_pdf(uploaded_pdf)
            if not input_text.strip():
                input_error = "No extractable text found in this PDF. It may be a scanned/image-based PDF."
            else:
                st.success(f"✅ Extracted text from `{uploaded_pdf.name}`")
                with st.expander("Preview extracted text"):
                    st.text(input_text[:2000] + ("..." if len(input_text) > 2000 else ""))
        except PDFReadError as e:
            input_error = f"Invalid or unreadable PDF: {e}"
        except Exception as e:
            input_error = f"Unexpected error reading PDF: {e}"

if input_error:
    st.error(f"⚠️ {input_error}")

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    word_count_preview = len(input_text.split()) if input_text else 0
    st.markdown(
        f"""<div class="metric-card"><div class="metric-value">{word_count_preview}</div>
        <div class="metric-label">Words Detected</div></div>""",
        unsafe_allow_html=True,
    )
with col_b:
    char_count_preview = len(input_text) if input_text else 0
    st.markdown(
        f"""<div class="metric-card"><div class="metric-value">{char_count_preview}</div>
        <div class="metric-label">Characters</div></div>""",
        unsafe_allow_html=True,
    )
with col_c:
    est_read = max(1, round(word_count_preview / 200)) if word_count_preview else 0
    st.markdown(
        f"""<div class="metric-card"><div class="metric-value">{est_read} min</div>
        <div class="metric-label">Est. Reading Time</div></div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
run_pipeline = st.button("🚀 Generate Summary &amp; Translation", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# VALIDATION
# ==============================================================================
MIN_WORDS = 30
MAX_CHARS = 60000  # guard against extremely large documents


def validate_input(text: str):
    if not text or not text.strip():
        return False, "Please provide some input text (paste, TXT, or PDF) before generating a summary."
    word_count = len(text.split())
    if word_count < MIN_WORDS:
        return False, f"Input too short. Please provide at least {MIN_WORDS} words (found {word_count})."
    if len(text) > MAX_CHARS:
        return False, (
            f"Document is very large ({len(text)} characters). "
            f"Please limit to {MAX_CHARS} characters, or split the document into smaller parts."
        )
    return True, None


# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
if run_pipeline:
    is_valid, validation_msg = validate_input(input_text)

    if not is_valid:
        st.error(f"⚠️ {validation_msg}")
    else:
        pipeline_start = time.time()
        stats = {}

        # ---------------- Extractive Summary ----------------
        with st.spinner("🔎 Generating extractive summary (Sumy LSA)..."):
            try:
                t0 = time.time()
                extractive_result = extractive_summary(
                    input_text, sentences_count=length_map[summary_length]
                )
                stats["extractive_time"] = round(time.time() - t0, 2)
            except ExtractiveSummaryError as e:
                st.error(f"⚠️ Extractive summarization failed: {e}")
                extractive_result = ""
                stats["extractive_time"] = 0.0
            except Exception as e:
                st.error(f"⚠️ Unexpected error during extractive summarization: {e}")
                extractive_result = ""
                stats["extractive_time"] = 0.0

        # ---------------- Gemini Abstractive Summary ----------------
        gemini_result = ""
        with st.spinner("✨ Generating Gemini AI abstractive summary..."):
            try:
                t0 = time.time()
                gemini_result = generate_gemini_summary(
                    input_text,
                    length=summary_length,
                    api_key=api_key_input.strip() if api_key_input else None,
                )
                stats["gemini_time"] = round(time.time() - t0, 2)
            except GeminiSummaryError as e:
                st.warning(
                    f"⚠️ Gemini API error: {e}. Falling back to the extractive summary for translation."
                )
                gemini_result = extractive_result
                stats["gemini_time"] = 0.0
            except Exception as e:
                st.warning(
                    f"⚠️ Unexpected Gemini error: {e}. Falling back to the extractive summary."
                )
                gemini_result = extractive_result
                stats["gemini_time"] = 0.0

        final_summary_for_translation = gemini_result if gemini_result.strip() else extractive_result

        # ---------------- Translation ----------------
        translated_result = ""
        with st.spinner(f"🌍 Translating summary into {target_lang_name.title()}..."):
            try:
                t0 = time.time()
                translated_result = translate_text(
                    final_summary_for_translation,
                    target_lang=LANGUAGES[target_lang_name],
                )
                stats["translation_time"] = round(time.time() - t0, 2)
            except TranslationError as e:
                st.error(f"⚠️ Translation failed: {e}")
                translated_result = ""
                stats["translation_time"] = 0.0
            except Exception as e:
                st.error(f"⚠️ Unexpected translation error: {e}")
                translated_result = ""
                stats["translation_time"] = 0.0

        total_time = round(time.time() - pipeline_start, 2)
        stats["total_time"] = total_time

        # ---------------- Compute Analytics ----------------
        original_words = len(input_text.split())
        extractive_words = len(extractive_result.split()) if extractive_result else 0
        gemini_words = len(gemini_result.split()) if gemini_result else 0
        translated_words = len(translated_result.split()) if translated_result else 0

        stats["original_words"] = original_words
        stats["extractive_words"] = extractive_words
        stats["gemini_words"] = gemini_words
        stats["translated_words"] = translated_words
        stats["compression_extractive"] = (
            round((1 - extractive_words / original_words) * 100, 1) if original_words and extractive_words else 0
        )
        stats["compression_gemini"] = (
            round((1 - gemini_words / original_words) * 100, 1) if original_words and gemini_words else 0
        )

        # ---------------- Persist to session state ----------------
        st.session_state["raw_text"] = input_text
        st.session_state["extractive_result"] = extractive_result
        st.session_state["gemini_result"] = gemini_result
        st.session_state["translated_result"] = translated_result
        st.session_state["target_lang_name"] = target_lang_name
        st.session_state["stats"] = stats
        st.session_state["pipeline_ran"] = True

        st.success(f"✅ Pipeline completed in {total_time}s")


# ==============================================================================
# RESULTS SECTION
# ==============================================================================
if st.session_state["pipeline_ran"]:
    stats = st.session_state["stats"]

    tab_summary, tab_translation, tab_analytics, tab_compare = st.tabs(
        ["📄 Summaries", "🌍 Translation", "📊 Analytics", "🔬 Comparison"]
    )

    # ---------------- TAB: SUMMARIES ----------------
    with tab_summary:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📌 Extractive Summary (Sumy LSA)</div>', unsafe_allow_html=True)
            st.write(st.session_state["extractive_result"] or "_No extractive summary generated._")
            st.markdown("</div>", unsafe_allow_html=True)
            if st.session_state["extractive_result"]:
                st.download_button(
                    "⬇️ Download Extractive Summary",
                    data=st.session_state["extractive_result"],
                    file_name="extractive_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">✨ Gemini AI Summary</div>', unsafe_allow_html=True)
            st.write(st.session_state["gemini_result"] or "_No Gemini summary generated._")
            st.markdown("</div>", unsafe_allow_html=True)
            if st.session_state["gemini_result"]:
                st.download_button(
                    "⬇️ Download Gemini Summary",
                    data=st.session_state["gemini_result"],
                    file_name="gemini_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    # ---------------- TAB: TRANSLATION ----------------
    with tab_translation:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card-title">🌍 Translated Summary '
            f'<span class="pill pill-info">{st.session_state["target_lang_name"].title()}</span></div>',
            unsafe_allow_html=True,
        )
        st.write(st.session_state["translated_result"] or "_No translation available._")
        st.markdown("</div>", unsafe_allow_html=True)
        if st.session_state["translated_result"]:
            st.download_button(
                "⬇️ Download Translated Summary",
                data=st.session_state["translated_result"],
                file_name=f"translated_summary_{st.session_state['target_lang_name']}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # ---------------- TAB: ANALYTICS ----------------
    with tab_analytics:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Key Metrics</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-value">{stats.get('original_words', 0)}</div>
                <div class="metric-label">Original Words</div></div>""",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-value">{stats.get('gemini_words', 0)}</div>
                <div class="metric-label">Gemini Summary Words</div></div>""",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-value">{stats.get('compression_gemini', 0)}%</div>
                <div class="metric-label">Compression (Gemini)</div></div>""",
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-value">{stats.get('total_time', 0)}s</div>
                <div class="metric-label">Total Processing Time</div></div>""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📈 Word Count Comparison</div>', unsafe_allow_html=True)
            bar_df = pd.DataFrame(
                {
                    "Stage": ["Original", "Extractive", "Gemini", "Translated"],
                    "Words": [
                        stats.get("original_words", 0),
                        stats.get("extractive_words", 0),
                        stats.get("gemini_words", 0),
                        stats.get("translated_words", 0),
                    ],
                }
            )
            fig_bar = px.bar(
                bar_df,
                x="Stage",
                y="Words",
                color="Stage",
                color_discrete_sequence=["#93c5fd", "#c4b5fd", "#f9a8d4", "#6ee7b7"],
                text="Words",
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f5f5fa",
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">⏱️ Processing Time Breakdown</div>', unsafe_allow_html=True)
            time_df = pd.DataFrame(
                {
                    "Stage": ["Extractive", "Gemini", "Translation"],
                    "Seconds": [
                        stats.get("extractive_time", 0),
                        stats.get("gemini_time", 0),
                        stats.get("translation_time", 0),
                    ],
                }
            )
            fig_pie = px.pie(
                time_df,
                names="Stage",
                values="Seconds",
                color_discrete_sequence=["#a78bfa", "#60a5fa", "#f472b6"],
                hole=0.55,
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f5f5fa",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📉 Compression Percentage</div>', unsafe_allow_html=True)
        comp_df = pd.DataFrame(
            {
                "Method": ["Extractive (Sumy LSA)", "Gemini AI"],
                "Compression %": [
                    stats.get("compression_extractive", 0),
                    stats.get("compression_gemini", 0),
                ],
            }
        )
        fig_comp = px.bar(
            comp_df,
            x="Method",
            y="Compression %",
            color="Method",
            color_discrete_sequence=["#c4b5fd", "#93c5fd"],
            text="Compression %",
        )
        fig_comp.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f5f5fa",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- TAB: COMPARISON ----------------
    with tab_compare:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔬 Side-by-Side Comparison Table</div>', unsafe_allow_html=True)

        comparison_table = pd.DataFrame(
            {
                "Metric": [
                    "Word Count",
                    "Character Count",
                    "Compression %",
                    "Processing Time (s)",
                ],
                "Original Document": [
                    stats.get("original_words", 0),
                    len(st.session_state["raw_text"]),
                    "—",
                    "—",
                ],
                "Extractive (Sumy LSA)": [
                    stats.get("extractive_words", 0),
                    len(st.session_state["extractive_result"]),
                    f"{stats.get('compression_extractive', 0)}%",
                    stats.get("extractive_time", 0),
                ],
                "Gemini AI Summary": [
                    stats.get("gemini_words", 0),
                    len(st.session_state["gemini_result"]),
                    f"{stats.get('compression_gemini', 0)}%",
                    stats.get("gemini_time", 0),
                ],
                "Translated Summary": [
                    stats.get("translated_words", 0),
                    len(st.session_state["translated_result"]),
                    "—",
                    stats.get("translation_time", 0),
                ],
            }
        )
        st.dataframe(comparison_table, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📦 Export Full Report</div>', unsafe_allow_html=True)
        report_text = (
            "Q38 SUMMARIZER + TRANSLATOR — FULL REPORT\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "=" * 60 + "\n\n"
            "ORIGINAL DOCUMENT (first 1000 chars):\n"
            f"{st.session_state['raw_text'][:1000]}\n\n"
            "EXTRACTIVE SUMMARY (Sumy LSA):\n"
            f"{st.session_state['extractive_result']}\n\n"
            "GEMINI AI SUMMARY:\n"
            f"{st.session_state['gemini_result']}\n\n"
            f"TRANSLATED SUMMARY ({st.session_state['target_lang_name'].title()}):\n"
            f"{st.session_state['translated_result']}\n\n"
            "STATISTICS:\n"
            f"{comparison_table.to_string(index=False)}\n"
        )
        st.download_button(
            "⬇️ Download Full Report (.txt)",
            data=report_text,
            file_name="Q38_summarizer_translator_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👆 Provide a document above and click **Generate Summary & Translation** to begin.")


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown(
    f"""
    <div class="app-footer">
        Built with ❤️ using <span>Streamlit</span>, <span>Google Gemini</span>,
        <span>Sumy</span> &amp; <span>Deep Translator</span><br>
        Q38 — Generative AI Lab · Summarizer + Translator Application<br>
        © {datetime.now().year} All rights reserved.
    </div>
    """,
    unsafe_allow_html=True,
)

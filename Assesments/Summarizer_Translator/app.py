"""
OmniBrief AI - Summarizer + Polyglot Translator Application
Question 38: Generative AI Lab
Architecture: Streamlit -> PDF Extraction -> Sumy LSA -> Google Gemini 3.6 Flash -> Deep Translator -> Plotly Analytics
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import local modular pipeline components
from extractive import generate_extractive_summary
from gemini_summary import generate_gemini_summary
from pdf_reader import extract_text_from_pdf, validate_pdf_content
from translator import get_supported_languages, translate_text

# Load environment variables
load_dotenv()

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="OmniBrief AI | Dual-Engine Summarizer & Translator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Glassmorphism & Modern AI Copilot CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #6366f1;
        --primary-hover: #4f46e5;
        --secondary: #06b6d4;
        --accent: #ec4899;
        --bg-dark: #090d16;
        --card-bg: rgba(17, 24, 39, 0.75);
        --card-border: rgba(255, 255, 255, 0.08);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Overall background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.10) 0%, transparent 40%),
                    linear-gradient(180deg, #090d16 0%, #0b1120 100%);
        color: var(--text-primary);
    }

    /* Glassmorphism Card System */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        margin-bottom: 20px;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow: 0 14px 35px -10px rgba(99, 102, 241, 0.2);
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 32px 28px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
    }

    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #ffffff 30%, #818cf8 70%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.5;
        max-width: 850px;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-right: 8px;
        margin-top: 12px;
    }

    .badge-primary {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.35);
    }

    .badge-secondary {
        background: rgba(6, 182, 212, 0.15);
        color: #67e8f9;
        border: 1px solid rgba(6, 182, 212, 0.35);
    }

    .badge-accent {
        background: rgba(236, 72, 153, 0.15);
        color: #f472b6;
        border: 1px solid rgba(236, 72, 153, 0.35);
    }

    /* Metric Card */
    .metric-box {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 2px;
    }

    .metric-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(15, 23, 42, 0.7);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        border: none;
        padding: 0 18px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b101c !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.02em;
        width: 100%;
        box-shadow: 0 6px 20px -3px rgba(99, 102, 241, 0.5);
        transition: all 0.3s ease;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(236, 72, 153, 0.6);
    }

    div.stDownloadButton > button {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }

    div.stDownloadButton > button:hover {
        background: rgba(99, 102, 241, 0.3) !important;
        border-color: #6366f1 !important;
    }

    /* Modern Footer */
    .app-footer {
        text-align: center;
        padding: 30px 10px 10px;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 50px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if "results_generated" not in st.session_state:
    st.session_state.results_generated = False
if "pipeline_data" not in st.session_state:
    st.session_state.pipeline_data = {}

# ==========================================
# 3. SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="background: linear-gradient(135deg, #6366f1, #ec4899); padding: 8px 12px; border-radius: 10px; font-size: 1.4rem;">⚡</div>
            <div>
                <h3 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #fff;">OmniBrief AI</h3>
                <p style="margin: 0; font-size: 0.75rem; color: #94a3b8;">GenAI Lab • Question 38</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🔑 API Configuration")

    # API Key Handling (Env or manual input)
    default_api_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=default_api_key,
        type="password",
        help="Reads from .env by default. You can also paste your Gemini API key here.",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Processing Parameters")

    # Input Method Selection
    input_source = st.radio(
        "Input Document Source",
        options=["Direct Text / Paste", "Upload TXT File", "Upload PDF Document"],
        index=0,
    )

    # Summary Length & Style
    summary_mode = st.selectbox(
        "AI Summary Granularity",
        options=["Concise (Executive)", "Standard (Balanced)", "Detailed (Comprehensive)"],
        index=1,
    )

    extractive_sentences = st.slider(
        "Extractive Sentences (Sumy LSA)",
        min_value=2,
        max_value=15,
        value=5,
        step=1,
        help="Number of salient sentences for the mathematical LSA extractive model.",
    )

    # Translation Options
    supported_langs = get_supported_languages()
    default_lang_idx = list(supported_langs.keys()).index("Spanish") if "Spanish" in supported_langs else 0
    target_language_name = st.selectbox(
        "Translate Summary To",
        options=list(supported_langs.keys()),
        index=default_lang_idx,
    )
    target_language_code = supported_langs[target_language_name]

    st.markdown("---")
    st.markdown("### 🧠 Model Specifications")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.6;">
            • <b>Abstractive:</b> Gemini 3.6 Flash<br>
            • <b>Extractive:</b> Sumy (LSA Algorithm)<br>
            • <b>Translation:</b> Deep Translator Engine<br>
            • <b>PDF Engine:</b> PyPDF2 Stream Processor
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================
# 4. HERO HEADER BANNER
# ==========================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">OmniBrief AI • Document Intelligence</div>
        <div class="hero-subtitle">
            Enterprise-grade document distillation pipeline. Ingest long-form text or PDF documents, generate dual-engine 
            (Extractive & Abstractive) summaries, and seamlessly translate synthesis into 30+ global languages with real-time compression analytics.
        </div>
        <div>
            <span class="badge badge-primary">✨ Google Gemini 3.6 Flash</span>
            <span class="badge badge-secondary">📐 Sumy LSA Extractive</span>
            <span class="badge badge-accent">🌐 Polyglot Neural Translation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 5. INPUT DOCUMENT WORKSPACE
# ==========================================
raw_document_text = ""

with st.container():
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="margin: 0 0 14px 0; color: #f8fafc; font-weight: 700;">📂 Document Ingestion Workspace</h4>
        """,
        unsafe_allow_html=True,
    )

    if input_source == "Direct Text / Paste":
        raw_document_text = st.text_area(
            "Paste your long-form text or article below:",
            height=240,
            placeholder="Paste comprehensive reports, research papers, news articles, or transcripts here (minimum 30 words recommended)...",
        )
    elif input_source == "Upload TXT File":
        uploaded_txt = st.file_uploader("Upload plain text file (.txt)", type=["txt"])
        if uploaded_txt is not None:
            try:
                raw_document_text = uploaded_txt.read().decode("utf-8")
                st.success(f"Successfully loaded '{uploaded_txt.name}' ({len(raw_document_text)} characters)")
            except Exception as e:
                st.error(f"Error reading TXT file: {str(e)}")
    elif input_source == "Upload PDF Document":
        uploaded_pdf = st.file_uploader("Upload PDF Document (.pdf)", type=["pdf"])
        if uploaded_pdf is not None:
            with st.spinner("Extracting text and structure from PDF..."):
                extracted_pdf_text, pdf_err = extract_text_from_pdf(uploaded_pdf)
                if pdf_err:
                    st.error(f"PDF Extraction Failed: {pdf_err}")
                else:
                    is_valid, validation_msg = validate_pdf_content(extracted_pdf_text)
                    if not is_valid:
                        st.warning(f"PDF Validation Warning: {validation_msg}")
                    else:
                        raw_document_text = extracted_pdf_text
                        st.success(f"Successfully extracted {len(raw_document_text.split())} words from PDF.")

    # Live Input Statistics
    if raw_document_text.strip():
        word_count = len(raw_document_text.split())
        char_count = len(raw_document_text)
        est_read_min = round(word_count / 200, 1)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""<div class="metric-box"><div class="metric-value">{word_count:,}</div><div class="metric-label">Source Words</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="metric-box"><div class="metric-value">{char_count:,}</div><div class="metric-label">Characters</div></div>""",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""<div class="metric-box"><div class="metric-value">{est_read_min} min</div><div class="metric-label">Est. Reading Time</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# Process Button
run_pipeline = st.button("🚀 Run Summarization & Translation Engine", use_container_width=True)

# ==========================================
# 6. PIPELINE EXECUTION ENGINE
# ==========================================
if run_pipeline:
    if not raw_document_text or len(raw_document_text.strip().split()) < 15:
        st.error("⚠️ Ingestion Error: Please provide at least 15 words of text or upload a valid document to summarize.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        start_total_time = time.time()
        pipeline_results = {
            "source_text": raw_document_text,
            "source_words": len(raw_document_text.split()),
            "source_chars": len(raw_document_text),
            "target_language_name": target_language_name,
        }

        # Step 1: Extractive Summarization (Sumy LSA)
        status_text.markdown("🔄 **Step 1/3:** Running Latent Semantic Analysis (LSA) Extractive Engine...")
        progress_bar.progress(25)
        t0 = time.time()
        extractive_summary, extractive_err = generate_extractive_summary(
            raw_document_text, sentence_count=extractive_sentences
        )
        extractive_time = round(time.time() - t0, 3)
        pipeline_results["extractive_summary"] = extractive_summary
        pipeline_results["extractive_error"] = extractive_err
        pipeline_results["extractive_time"] = extractive_time
        pipeline_results["extractive_words"] = len(extractive_summary.split()) if extractive_summary else 0

        # Step 2: Abstractive AI Summarization (Gemini 3.6 Flash)
        status_text.markdown("🤖 **Step 2/3:** Synthesizing with Google Gemini 3.6 Flash...")
        progress_bar.progress(55)
        t1 = time.time()
        gemini_summary, key_takeaways, gemini_err = generate_gemini_summary(
            text=raw_document_text,
            api_key=api_key_input,
            summary_length=summary_mode,
        )
        gemini_time = round(time.time() - t1, 3)
        pipeline_results["gemini_summary"] = gemini_summary
        pipeline_results["key_takeaways"] = key_takeaways
        pipeline_results["gemini_error"] = gemini_err
        pipeline_results["gemini_time"] = gemini_time
        pipeline_results["gemini_words"] = len(gemini_summary.split()) if gemini_summary else 0

        # Step 3: Polyglot Neural Translation
        status_text.markdown(f"🌐 **Step 3/3:** Translating synthesized intelligence into {target_language_name}...")
        progress_bar.progress(85)
        t2 = time.time()

        # Translate the best available summary (Gemini preferred, Extractive fallback)
        text_to_translate = gemini_summary if (gemini_summary and not gemini_err) else extractive_summary
        translated_text, translation_err = translate_text(
            text=text_to_translate,
            target_language_code=target_language_code,
        )
        translation_time = round(time.time() - t2, 3)
        pipeline_results["translated_summary"] = translated_text
        pipeline_results["translation_error"] = translation_err
        pipeline_results["translation_time"] = translation_time
        pipeline_results["translated_words"] = len(translated_text.split()) if translated_text else 0

        # Finalize Metrics
        total_pipeline_time = round(time.time() - start_total_time, 2)
        pipeline_results["total_time"] = total_pipeline_time

        # Compression calculation
        best_summary_words = pipeline_results["gemini_words"] if pipeline_results["gemini_words"] > 0 else pipeline_results["extractive_words"]
        compression_ratio = 0
        if pipeline_results["source_words"] > 0 and best_summary_words > 0:
            compression_ratio = round(
                (1 - (best_summary_words / pipeline_results["source_words"])) * 100, 1
            )
        pipeline_results["compression_ratio"] = max(0, compression_ratio)

        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

        # Store in session state
        st.session_state.pipeline_data = pipeline_results
        st.session_state.results_generated = True
        st.toast("Intelligence brief successfully generated!", icon="⚡")

# ==========================================
# 7. RESULTS & ANALYTICS DASHBOARD
# ==========================================
if st.session_state.results_generated and st.session_state.pipeline_data:
    data = st.session_state.pipeline_data

    # High-level Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(
            f"""<div class="metric-box"><div class="metric-value">{data['source_words']:,}</div><div class="metric-label">Source Words</div></div>""",
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""<div class="metric-box"><div class="metric-value" style="color:#818cf8;">{data['gemini_words']:,}</div><div class="metric-label">Gemini Words</div></div>""",
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""<div class="metric-box"><div class="metric-value" style="color:#67e8f9;">{data['extractive_words']:,}</div><div class="metric-label">LSA Words</div></div>""",
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""<div class="metric-box"><div class="metric-value" style="color:#34d399;">{data['compression_ratio']}%</div><div class="metric-label">Compression</div></div>""",
            unsafe_allow_html=True,
        )
    with m_col5:
        st.markdown(
            f"""<div class="metric-box"><div class="metric-value" style="color:#f472b6;">{data['total_time']}s</div><div class="metric-label">Total Latency</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # Tabbed Display
    tab_gemini, tab_translate, tab_extractive, tab_analytics = st.tabs(
        [
            "🤖 Gemini AI Summary",
            f"🌐 Polyglot Translation ({data['target_language_name']})",
            "📐 Extractive LSA Summary",
            "📊 Analytics & Benchmarks",
        ]
    )

    # ---------------- TAB 1: GEMINI ABSTRACTIVE ----------------
    with tab_gemini:
        if data.get("gemini_error"):
            st.error(f"Gemini Summarization Issue: {data['gemini_error']}")
            st.info("💡 Make sure your Gemini API key is valid and configured in the sidebar or `.env` file.")
        else:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <h4 style="margin:0; color:#818cf8; font-weight:700;">✨ Google Gemini 3.6 Flash Abstractive Brief</h4>
                        <span style="font-size:0.8rem; color:#94a3b8;">Latency: {data['gemini_time']}s</span>
                    </div>
                    <div style="font-size:1.02rem; line-height:1.7; color:#e2e8f0; white-space: pre-wrap;">
{data['gemini_summary']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Key Bullet Takeaways if available
            if data.get("key_takeaways"):
                with st.expander("📌 Core Executive Takeaways", expanded=True):
                    for idx, point in enumerate(data["key_takeaways"], 1):
                        st.markdown(f"**{idx}.** {point}")

            # Export Options
            dl_col1, dl_col2 = st.columns([1, 4])
            with dl_col1:
                st.download_button(
                    label="📥 Download Summary (TXT)",
                    data=data["gemini_summary"],
                    file_name=f"gemini_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    # ---------------- TAB 2: TRANSLATION ----------------
    with tab_translate:
        if data.get("translation_error"):
            st.error(f"Translation Error: {data['translation_error']}")
        else:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <h4 style="margin:0; color:#f472b6; font-weight:700;">🌐 Neural Translation ({data['target_language_name']})</h4>
                        <span style="font-size:0.8rem; color:#94a3b8;">Latency: {data['translation_time']}s</span>
                    </div>
                    <div style="font-size:1.05rem; line-height:1.8; color:#f1f5f9; white-space: pre-wrap;">
{data['translated_summary']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            dl_t1, dl_t2 = st.columns([1, 4])
            with dl_t1:
                st.download_button(
                    label=f"📥 Download ({data['target_language_name']})",
                    data=data["translated_summary"],
                    file_name=f"translated_{data['target_language_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    # ---------------- TAB 3: EXTRACTIVE (SUMY LSA) ----------------
    with tab_extractive:
        if data.get("extractive_error"):
            st.error(f"Extractive Summarization Error: {data['extractive_error']}")
        else:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <h4 style="margin:0; color:#67e8f9; font-weight:700;">📐 Latent Semantic Analysis (LSA) Extraction</h4>
                        <span style="font-size:0.8rem; color:#94a3b8;">Latency: {data['extractive_time']}s</span>
                    </div>
                    <div style="font-size:1.0rem; line-height:1.7; color:#e2e8f0; white-space: pre-wrap;">
{data['extractive_summary']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            dl_e1, dl_e2 = st.columns([1, 4])
            with dl_e1:
                st.download_button(
                    label="📥 Download LSA Summary",
                    data=data["extractive_summary"],
                    file_name=f"lsa_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    # ---------------- TAB 4: ANALYTICS & BENCHMARKS ----------------
    with tab_analytics:
        st.markdown("#### 📊 Comparative Performance & Compression Metrics")

        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            # Word Count Bar Chart
            df_words = pd.DataFrame(
                {
                    "Pipeline Stage": ["Source Text", "Sumy Extractive", "Gemini AI", f"Translated ({data['target_language_name']})"],
                    "Word Count": [data["source_words"], data["extractive_words"], data["gemini_words"], data["translated_words"]],
                    "Color": ["#94a3b8", "#06b6d4", "#6366f1", "#ec4899"],
                }
            )

            fig_bar = px.bar(
                df_words,
                x="Pipeline Stage",
                y="Word Count",
                color="Pipeline Stage",
                color_discrete_sequence=["#94a3b8", "#06b6d4", "#6366f1", "#ec4899"],
                text="Word Count",
                title="Word Count Across Processing Stages",
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(17, 24, 39, 0.4)",
                plot_bgcolor="rgba(17, 24, 39, 0.4)",
                showlegend=False,
                font=dict(family="Plus Jakarta Sans", color="#f8fafc"),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_c2:
            # Latency Breakdown Donut Chart
            latency_labels = ["Sumy LSA", "Gemini 3.6 Flash", "Deep Translator"]
            latency_values = [data["extractive_time"], data["gemini_time"], data["translation_time"]]

            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=latency_labels,
                        values=latency_values,
                        hole=0.55,
                        marker_colors=["#06b6d4", "#6366f1", "#ec4899"],
                    )
                ]
            )
            fig_donut.update_layout(
                title="Execution Latency Breakdown (Seconds)",
                template="plotly_dark",
                paper_bgcolor="rgba(17, 24, 39, 0.4)",
                plot_bgcolor="rgba(17, 24, 39, 0.4)",
                font=dict(family="Plus Jakarta Sans", color="#f8fafc"),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Comparative Summary Table
        st.markdown("#### 📋 Detailed Pipeline Diagnostic Matrix")
        df_comparison = pd.DataFrame(
            [
                {
                    "Metric": "Word Count",
                    "Source Document": f"{data['source_words']:,} words",
                    "Sumy LSA Extractive": f"{data['extractive_words']:,} words",
                    "Gemini AI Abstractive": f"{data['gemini_words']:,} words",
                    "Polyglot Translation": f"{data['translated_words']:,} words",
                },
                {
                    "Metric": "Processing Time",
                    "Source Document": "N/A (Ingestion)",
                    "Sumy LSA Extractive": f"{data['extractive_time']} s",
                    "Gemini AI Abstractive": f"{data['gemini_time']} s",
                    "Polyglot Translation": f"{data['translation_time']} s",
                },
                {
                    "Metric": "Compression vs Source",
                    "Source Document": "0% (Baseline)",
                    "Sumy LSA Extractive": f"{round((1 - (data['extractive_words']/max(1, data['source_words'])))*100, 1)}%",
                    "Gemini AI Abstractive": f"{data['compression_ratio']}%",
                    "Polyglot Translation": f"{round((1 - (data['translated_words']/max(1, data['source_words'])))*100, 1)}%",
                },
                {
                    "Metric": "Engine Paradigm",
                    "Source Document": "Raw Text / PDF",
                    "Sumy LSA Extractive": "Unsupervised Mathematical LSA",
                    "Gemini AI Abstractive": "Deep LLM (Gemini 3.6 Flash)",
                    "Polyglot Translation": "Google Neural Translation",
                },
            ]
        )
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

# ==========================================
# 8. FOOTER
# ==========================================
st.markdown(
    """
    <div class="app-footer">
        <b>Q38_Summarizer_Translator</b> • Generative AI Lab Question 38 • Powered by Google Gemini & Streamlit<br>
        Built with Streamlit, Google GenAI SDK, Sumy LSA, PyPDF2, and Deep Translator.
    </div>
    """,
    unsafe_allow_html=True,
)
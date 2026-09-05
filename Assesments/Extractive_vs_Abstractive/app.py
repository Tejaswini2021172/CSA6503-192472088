import streamlit as st
import time
import pandas as pd
from extractive import extractive_summary
from gemini_summary import abstractive_summary

def split_text(text, chunk_size=5000):
    """
    Split large text into smaller chunks.
    """
    return [
        text[i:i+chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

/* ===============================
   Main Background
=================================*/
.stApp{
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e3a8a 45%,
        #312e81 100%
    );
    color:white;
}

/* ===============================
   Hero Title
=================================*/
.main-title{
    font-size:48px;
    font-weight:800;
    text-align:center;
    color:white;
    margin-top:10px;
}

.sub-title{
    text-align:center;
    font-size:20px;
    color:#d1d5db;
    margin-bottom:30px;
}

/* ===============================
   Glass Cards
=================================*/
.block-container{

    padding-top:2rem;
    padding-left:3rem;
    padding-right:3rem;

}

div[data-testid="stVerticalBlock"]{
    border-radius:18px;
}

/* ===============================
   Input Box
=================================*/

textarea{
    background:rgba(255,255,255,0.08)!important;
    color:white!important;

    border-radius:15px!important;

    border:1px solid rgba(255,255,255,0.15)!important;
}

/* ===============================
   Buttons
=================================*/

.stButton>button{

background:linear-gradient(90deg,#2563eb,#7c3aed);

color:white;

font-size:18px;

font-weight:bold;

border-radius:12px;

height:55px;

border:none;

transition:0.3s;

}

.stButton>button:hover{

transform:scale(1.03);

box-shadow:0px 0px 20px #60a5fa;

}

/* ===============================
   Download Buttons
=================================*/

.stDownloadButton>button{

background:#16a34a;

color:white;

border-radius:10px;

font-weight:bold;

}

/* ===============================
   Metric Cards
=================================*/

div[data-testid="metric-container"]{

background:rgba(255,255,255,0.08);

border-radius:15px;

padding:18px;

border:1px solid rgba(255,255,255,0.12);

box-shadow:0px 5px 15px rgba(0,0,0,.3);

}

/* ===============================
   Sidebar
=================================*/

section[data-testid="stSidebar"]{

background:linear-gradient(

180deg,

#111827,

#1f2937

);

}

/* ===============================
   Expanders
=================================*/

.streamlit-expanderHeader{

font-size:18px;

font-weight:bold;

color:white;

}

/* ===============================
   Table
=================================*/

table{

border-radius:15px;

overflow:hidden;

}

/* ===============================
   Footer
=================================*/

.footer{

text-align:center;

color:#d1d5db;

font-size:15px;

margin-top:30px;

}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div style='
padding:35px;
border-radius:20px;
background:linear-gradient(90deg,#2563eb,#7c3aed);
text-align:center;
box-shadow:0px 10px 30px rgba(0,0,0,.35);
'>

<h1 style='color:white;font-size:48px;margin-bottom:5px;'>
🧠 AI Text Summarizer
</h1>

<p style='color:white;font-size:20px;'>
Compare Extractive vs Abstractive Summaries using Google Gemini AI
</p>

</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙ Settings")

summary_length = st.sidebar.selectbox(
    "Summary Length",
    ["Short", "Medium", "Long"]
)
st.sidebar.markdown("---")

st.sidebar.subheader("🤖 AI Models")

st.sidebar.success("""
**Extractive**
- LSA (Sumy)

**Abstractive**
- Gemini 3.6 Flash
""")

input_type = st.sidebar.radio(
    "Input Type",
    ["Paste Text", "Upload File"]
)

# -----------------------------
# Input Section
# -----------------------------
st.header("📄 Input")

text = ""
uploaded_file = None

if input_type == "Paste Text":

    text = st.text_area(
        "Paste your article here",
        height=250
    )

else:

    uploaded_file = st.file_uploader(
    "📄 Upload Research Paper, Report or Article",
    type=["pdf", "txt"],
    help="Supported formats: PDF, TXT"
    )


    if uploaded_file is not None:

       if uploaded_file.type == "text/plain":

          text = uploaded_file.read().decode("utf-8")

       elif uploaded_file.type == "application/pdf":

          from pdf_reader import extract_pdf_text

          text = extract_pdf_text(uploaded_file)

       st.success("✅ File uploaded successfully!")

       # File Information
       st.write(f"**📄 File Name:** {uploaded_file.name}")
       st.write(f"**📦 File Size:** {round(uploaded_file.size/1024,2)} KB")
       # Document Statistics (Step 7)
       st.markdown("---")
       c1, c2, c3 = st.columns(3)
       c1.metric("📄 Characters", len(text))
       c2.metric("📝 Words", len(text.split()))
       c3.metric("📦 File Size", f"{round(uploaded_file.size/1024,2)} KB")

    else:
       st.info("Please upload a TXT or PDF file.")
# -----------------------------
# Generate Button
# -----------------------------
generate = st.button(
    "🚀 Generate Summary",
    use_container_width=True
)

# -----------------------------
# Output
# -----------------------------
if generate:
    start = time.time()

    if input_type == "Paste Text":

        if not text.strip():
            st.error("Please enter some text.")
            st.stop()

    st.success("✅ Summaries generated successfully!")

    # Create two columns
    col1, col2 = st.columns(2)

    # -----------------------------
    # Extractive Summary
    # -----------------------------
    with col1:

        st.subheader("📌 Extractive Summary")

        summary = extractive_summary(text, summary_length)

        with st.expander("📌 View Extractive Summary", expanded=True):
            st.write(summary)

        st.caption(f"Words : {len(summary.split())}")

        st.download_button(
            label="📥 Download Extractive Summary",
            data=summary,
            file_name="extractive_summary.txt",
            mime="text/plain"
   )


    # -----------------------------
    # Abstractive Summary
    # -----------------------------
    with col2:

        st.subheader("🤖 Abstractive Summary")

        with st.spinner("Generating AI Summary..."):

            chunks = split_text(text)

            summaries = []

            progress = st.progress(0)

            for i, chunk in enumerate(chunks):

               summaries.append(
                   abstractive_summary(chunk, summary_length)
                )
            progress.progress((i + 1) / len(chunks))

            progress.empty()

            ai_summary = "\n\n".join(summaries)

            end = time.time()
            st.success(f"⏱ Generated in {end-start:.2f} seconds")

        with st.expander("🤖 View Abstractive Summary", expanded=True):
            st.write(ai_summary)

        st.caption(f"Words : {len(ai_summary.split())}")

        st.download_button(
           label="📥 Download Abstractive Summary",
           data=ai_summary,
           file_name="abstractive_summary.txt",
           mime="text/plain"
        )
    # -----------------------------
    # Statistics
    # -----------------------------
    st.divider()

    st.subheader("📊 Statistics")

    original_words = len(text.split())
    extractive_words = len(summary.split())
    abstractive_words = len(ai_summary.split())

    extractive_compression = round(
        ((original_words - extractive_words) / original_words) * 100,
          1
    )

    abstractive_compression = round(
        ((original_words - abstractive_words) / original_words) * 100,
          1
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Original", original_words)
    c2.metric("Extractive", extractive_words)
    c3.metric("Abstractive", abstractive_words)
    c4.metric("Extractive Compression", f"{extractive_compression}%")
    c5.metric("Abstractive Compression", f"{abstractive_compression}%")

    comparison = pd.DataFrame({
        "Feature": [
                "Technique",
                "Words",
                "Compression"
        ],
        "Extractive": [
               "LSA (Sumy)",
                extractive_words,
                f"{extractive_compression}%"
        ],
       "Abstractive": [
            "Gemini 3.6 Flash",
            abstractive_words,
            f"{abstractive_compression}%"
        ]
   })

    st.subheader("📋 Comparison")

    st.table(comparison)

    st.markdown("---")

    st.markdown("""
      <div style='text-align:center;color:gray'>
      Developed using ❤️ Streamlit | Sumy | Google Gemini API
      </div>
     """, unsafe_allow_html=True
)
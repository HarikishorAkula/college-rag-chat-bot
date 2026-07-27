import os
import re
from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape
from groq import Groq
import streamlit as st
from dotenv import load_dotenv

from pypdf import PdfReader
from docx import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)


# -----------------------------
# 1. PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Lumen — AI Notes Companion",
    page_icon="🔆",
    layout="wide"
)


# -----------------------------
# 2. CUSTOM UI CSS  —  "Lumen" desk-lamp theme
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --ink-dark: #0b1220;
        --ink: #10182a;
        --ink-light: #1b2436;
        --ink-lighter: #232f47;
        --text: #eef1f7;
        --muted: #93a1b8;
        --amber: #f5b754;
        --amber-dim: #d99a3d;
        --teal: #5eead4;
        --teal-dim: #38beac;
    }

    .stApp {
        background:
            radial-gradient(ellipse 900px 460px at 50% -8%, rgba(245, 183, 84, 0.20), transparent 62%),
            radial-gradient(ellipse 600px 400px at 85% 15%, rgba(94, 234, 212, 0.06), transparent 60%),
            linear-gradient(180deg, var(--ink-dark) 0%, var(--ink) 100%);
        background-attachment: fixed;
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .main-title {
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-size: 48px;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 4px;
        letter-spacing: 0.3px;
        text-shadow: 0px 0px 30px rgba(245, 183, 84, 0.35);
    }

    .lumen-glow-bar {
        display: block;
        width: 220px;
        height: 4px;
        margin: 10px auto 16px auto;
        border-radius: 4px;
        background: linear-gradient(90deg, transparent, var(--amber), transparent);
        box-shadow: 0px 0px 18px 2px rgba(245, 183, 84, 0.55);
    }

    .sub-title {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        color: var(--muted);
        margin-bottom: 32px;
        font-weight: 500;
        letter-spacing: 0.2px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
        font-family: 'Playfair Display', serif;
    }

    p, li, label, span, div {
        color: var(--text) !important;
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11,18,32,0.98), rgba(16,24,42,0.98)) !important;
        border-right: 1px solid rgba(245, 183, 84, 0.15);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: var(--text) !important;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(27, 36, 54, 0.75) !important;
        border: 1.5px solid rgba(245, 183, 84, 0.35) !important;
        border-radius: 14px !important;
        padding: 12px !important;
    }

    div[data-testid="stFileUploader"] * {
        color: var(--text) !important;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(16, 24, 42, 0.8) !important;
        border: 1px dashed rgba(94, 234, 212, 0.35) !important;
        border-radius: 12px !important;
    }

    div[data-testid="stFileUploaderDropzone"] * {
        color: var(--text) !important;
    }

    div[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, var(--amber), var(--amber-dim)) !important;
        color: #1a1206 !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 700 !important;
    }

    div[data-testid="stFileUploaderDropzone"] button * {
        color: #1a1206 !important;
    }

    div[data-testid="stFileUploaderFile"] {
        background: rgba(11, 18, 32, 0.9) !important;
        border: 1px solid rgba(245, 183, 84, 0.25) !important;
        border-radius: 10px !important;
        padding: 8px !important;
    }

    div[data-testid="stFileUploaderFile"] * {
        color: var(--text) !important;
    }

    div[data-testid="stFileUploaderFileName"] {
        color: var(--text) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stFileUploaderFileSize"] {
        color: var(--teal) !important;
    }

    div[data-testid="stFileUploaderFile"] button {
        background: transparent !important;
        color: #f08080 !important;
        border-radius: 50% !important;
    }

    div[data-baseweb="select"] > div {
        background: rgba(16, 24, 42, 0.9) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(245, 183, 84, 0.3) !important;
    }

    div[data-baseweb="select"] span {
        color: var(--text) !important;
    }

    div[data-baseweb="popover"] * {
        background-color: var(--ink) !important;
        color: var(--text) !important;
    }

    input {
        background: rgba(16, 24, 42, 0.9) !important;
        color: var(--text) !important;
        border: 1px solid rgba(245, 183, 84, 0.3) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
    }

    input::placeholder {
        color: var(--muted) !important;
    }

    textarea {
        background: rgba(16, 24, 42, 0.9) !important;
        color: var(--text) !important;
        border: 1px solid rgba(245, 183, 84, 0.3) !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        border: none;
        background: linear-gradient(135deg, var(--amber), var(--amber-dim));
        color: #1a1206 !important;
        padding: 10px 24px;
        transition: 0.2s;
        box-shadow: 0px 4px 20px rgba(245, 183, 84, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 26px rgba(245, 183, 84, 0.4);
        color: #1a1206 !important;
    }

    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        background: linear-gradient(135deg, var(--teal), var(--teal-dim));
        color: #06231d !important;
        border: none;
        padding: 10px 24px;
        box-shadow: 0px 4px 20px rgba(94, 234, 212, 0.25);
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 26px rgba(94, 234, 212, 0.4);
        color: #06231d !important;
    }

    .glass-card {
        background: rgba(27, 36, 54, 0.6);
        color: var(--text);
        padding: 24px 26px;
        border-radius: 14px;
        border-left: 4px solid var(--amber);
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0px 8px 28px rgba(0,0,0,0.25);
    }

    .glass-card * {
        color: var(--text) !important;
    }

    .answer-card {
        background: rgba(16, 24, 42, 0.75);
        color: var(--text);
        padding: 26px 28px;
        border-radius: 14px;
        border-left: 4px solid var(--amber);
        margin-top: 20px;
        margin-bottom: 20px;
        line-height: 1.7;
        box-shadow: 0px 8px 30px rgba(0,0,0,0.3);
    }

    .answer-card h2 {
        font-family: 'Playfair Display', serif;
        color: var(--amber) !important;
        margin-top: 0;
        font-size: 22px;
    }

    .answer-card * {
        color: var(--text) !important;
    }

    .answer-text {
        font-size: 16px;
        line-height: 1.85;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif;
    }

    .answer-text b {
        color: var(--amber) !important;
        font-weight: 700 !important;
        font-size: 17px;
    }

    /* --- Empty / "not in notes" state --- */
    .noinfo-card {
        background: rgba(27, 36, 54, 0.55);
        color: var(--muted);
        padding: 26px 28px;
        border-radius: 14px;
        border-left: 4px solid var(--muted);
        margin-top: 20px;
        margin-bottom: 20px;
        text-align: left;
    }

    .noinfo-card h2 {
        font-family: 'Playfair Display', serif;
        color: var(--muted) !important;
        margin-top: 0;
        font-size: 20px;
    }

    .noinfo-card p, .noinfo-card * {
        color: var(--muted) !important;
    }

    .start-card {
        background: rgba(27, 36, 54, 0.55);
        color: var(--text);
        padding: 34px 24px;
        border-radius: 14px;
        border: 1px solid rgba(245, 183, 84, 0.2);
        margin-top: 25px;
        text-align: center;
    }

    .start-card * {
        color: var(--text) !important;
    }

    .start-card h3 {
        font-family: 'Playfair Display', serif;
        color: var(--amber) !important;
    }

    div[data-testid="column"] * {
        color: var(--text) !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(16, 24, 42, 0.7) !important;
        border: 1px solid rgba(245, 183, 84, 0.2) !important;
        border-radius: 12px !important;
    }

    div[data-testid="stExpander"] * {
        color: var(--text) !important;
    }

    .streamlit-expanderHeader {
        color: var(--text) !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stAlert {
        background: rgba(16, 24, 42, 0.9) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(245, 183, 84, 0.25) !important;
    }

    .stAlert * {
        color: var(--text) !important;
    }

    hr {
        border: none;
        border-top: 1px solid rgba(245, 183, 84, 0.18) !important;
    }

    .stCaptionContainer, .stCaptionContainer * {
        color: var(--teal) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* --- Notebook-style chat history cards --- */
    .note-card {
        background: rgba(27, 36, 54, 0.5);
        border-left: 3px solid var(--teal);
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 4px;
    }

    .note-q {
        font-family: 'Playfair Display', serif;
        color: var(--teal) !important;
        font-size: 17px;
        margin-bottom: 8px;
    }

    .note-a {
        font-family: 'Inter', sans-serif;
        color: var(--text) !important;
        font-size: 15px;
        line-height: 1.7;
        border-left: 2px solid var(--amber);
        padding-left: 12px;
    }

    /* --- Status badge --- */
    .status-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: 6px;
        background: rgba(245, 183, 84, 0.12);
        border: 1px solid rgba(245, 183, 84, 0.3);
        color: var(--amber) !important;
        margin: 2px 4px 2px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# 3. LOAD GROQ API KEY
# -----------------------------
def get_groq_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        load_dotenv()
        return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()


@st.cache_resource
def get_groq_client(api_key):
    return Groq(api_key=api_key)


# -----------------------------
# 4. EXTRACT TEXT FROM PDF
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PdfReader(uploaded_file)

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            text += f"\n\n--- Source: {uploaded_file.name}, Page: {page_number} ---\n"
            text += page_text

    return text


# -----------------------------
# 5. EXTRACT TEXT FROM DOCX
# -----------------------------
def extract_text_from_docx(uploaded_file):
    text = ""
    doc = Document(uploaded_file)

    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"

    return f"\n\n--- Source: {uploaded_file.name} ---\n{text}"


# -----------------------------
# 6. EXTRACT TEXT FROM TXT
# -----------------------------
def extract_text_from_txt(uploaded_file):
    file_bytes = uploaded_file.read()
    text = file_bytes.decode("utf-8", errors="ignore")

    return f"\n\n--- Source: {uploaded_file.name} ---\n{text}"


# -----------------------------
# 7. EXTRACT TEXT FROM ALL FILES
# -----------------------------
def extract_text_from_uploaded_files(uploaded_files):
    full_text = ""

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".pdf"):
            full_text += extract_text_from_pdf(uploaded_file)

        elif file_name.endswith(".docx"):
            full_text += extract_text_from_docx(uploaded_file)

        elif file_name.endswith(".txt"):
            full_text += extract_text_from_txt(uploaded_file)

        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")

    return full_text


# -----------------------------
# 8. SPLIT TEXT INTO CHUNKS
# -----------------------------
def create_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    return chunks


# -----------------------------
# 9. CREATE VECTOR STORE
# -----------------------------
@st.cache_resource
def load_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def create_vector_store(chunks):
    embeddings = load_embedding_model()

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vector_store


# -----------------------------
# 10. ASK GROQ LLM
# -----------------------------
NO_INFO_TEXT = "No info"


def ask_groq(question, context, answer_style):
    if not GROQ_API_KEY:
        return "Groq API key not found. Please add GROQ_API_KEY."

    client = get_groq_client(GROQ_API_KEY)

    prompt = f"""
You are a College Notes Assistant.

You must answer mainly from the uploaded document context.

Priority rules:

1. First preference is always the uploaded document context.
2. If the answer is clearly available in the context:
   - Answer only using the document context.
   - Do not add extra outside information.
3. If the question is related to the uploaded document topic, but the exact answer is not fully present in the context:
   - Start with this line:
     This exact answer was not found in the uploaded notes, but here is a general explanation:
   - Then give a simple general explanation.
4. If the question is not related to the uploaded document:
   - Reply only:
     {NO_INFO_TEXT}
5. Do not use markdown symbols like **, *, ###.
6. For headings, write them clearly like:
   Definition:
   Role:
   Example:
7. Keep headings and side headings short.
8. Use simple student-friendly language.
9. Format the answer according to this answer style: {answer_style}

Uploaded Document Context:
{context}

Student Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# 11. RETRIEVE RELEVANT CHUNKS WITH SCORE
# -----------------------------
def retrieve_relevant_docs(vector_store, question, k=6, max_score=1.35):
    """
    FAISS similarity_search_with_score returns lower score for better match.
    If best score is too high, question is treated as not related to uploaded notes.

    max_score:
    - lower value = stricter document matching
    - higher value = more flexible document matching
    """
    results = vector_store.similarity_search_with_score(question, k=k)

    if not results:
        return [], False, None

    best_score = results[0][1]

    if best_score > max_score:
        return [], False, best_score

    docs = [doc for doc, score in results]
    return docs, True, best_score


# -----------------------------
# 12. CACHE KEY FUNCTION
# -----------------------------
def make_cache_key(question, answer_style):
    clean_question = question.strip().lower()
    clean_style = answer_style.strip().lower()
    return f"{clean_question}__{clean_style}"


# -----------------------------
# 13. PDF CREATION FUNCTIONS
# -----------------------------
def clean_text_for_pdf(text):
    if text is None:
        return ""

    text = str(text)

    # Convert markdown bold to reportlab bold
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

    # Remove remaining star marks
    text = text.replace("* ", "")
    text = text.replace("*", "")

    text = escape(text)
    text = text.replace("&lt;b&gt;", "<b>")
    text = text.replace("&lt;/b&gt;", "</b>")
    text = text.replace("\n", "<br/>")

    return text


def clean_text_for_html(text):
    if text is None:
        return ""

    text = str(text)

    # Convert markdown bold **text** into HTML bold
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

    # Remove remaining stars
    text = text.replace("* ", "")
    text = text.replace("*", "")

    text = escape(text)
    text = text.replace("&lt;b&gt;", "<b>")
    text = text.replace("&lt;/b&gt;", "</b>")

    # Make lines ending with ":" bold headings
    lines = text.split("\n")
    formatted_lines = []

    for line in lines:
        clean_line = line.strip()

        if clean_line.endswith(":") and len(clean_line) <= 60:
            formatted_lines.append(f"<b>{clean_line}</b>")
        else:
            formatted_lines.append(line)

    text = "<br>".join(formatted_lines)

    return text


def get_unique_chat_history(chat_history):
    unique_chats = []
    seen_questions = set()

    for chat in chat_history:
        question = chat.get("question", "").strip().lower()

        if question and question not in seen_questions:
            unique_chats.append(chat)
            seen_questions.add(question)

    return unique_chats


def create_chat_pdf(chat_history, answer_style, file_names):
    unique_chat_history = get_unique_chat_history(chat_history)

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=18
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=25
    )

    question_style = ParagraphStyle(
        "QuestionStyle",
        parent=styles["Heading3"],
        fontSize=12,
        textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=12,
        spaceAfter=8
    )

    meta_style = ParagraphStyle(
        "MetaStyle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#000000"),
        spaceAfter=10
    )

    story = []

    story.append(Paragraph("Lumen — AI Notes Companion", title_style))
    story.append(Paragraph("Questions and Answers Report", subtitle_style))
    story.append(Paragraph(f"<b>Total Questions:</b> {len(unique_chat_history)}", meta_style))

    story.append(Spacer(1, 12))

    for index, chat in enumerate(unique_chat_history, start=1):
        question = clean_text_for_pdf(chat.get("question", ""))
        answer = clean_text_for_pdf(chat.get("answer", ""))

        story.append(Paragraph(f"Question {index}: {question}", question_style))
        story.append(Paragraph(f"<b>Answer:</b><br/>{answer}"))
        story.append(Spacer(1, 12))

    doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


# -----------------------------
# 14. INITIALIZE SESSION STATE
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}

if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = []


# -----------------------------
# 15. HEADER UI
# -----------------------------
st.markdown(
    "<div class='main-title'>🔆 Lumen — AI Notes Companion</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<span class='lumen-glow-bar'></span>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Upload your notes, ask anything, and export the full session as a PDF.</div>",
    unsafe_allow_html=True
)


# -----------------------------
# 16. SIDEBAR UI
# -----------------------------
with st.sidebar:
    st.markdown("## 📤 Upload Notes")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT notes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    answer_style = st.selectbox(
        "Choose answer style",
        [
            "Simple explanation",
            "Detailed explanation",
            "Exam point of view",
            "Short notes",
            "Important points"
        ]
    )

    process_button = st.button("🚀 Process Notes")

    st.markdown("---")
    st.markdown("## 🗂️ Status")

    if st.session_state.vector_store is not None:
        st.success("Notes are ready.")
        if st.session_state.processed_file_names:
            badges = "".join(
                f"<span class='status-badge'>📄 {file_name}</span>"
                for file_name in st.session_state.processed_file_names
            )
            st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
    else:
        st.info("Upload and process notes first.")

    st.markdown("---")

    if st.session_state.chat_history:
        pdf_data_sidebar = create_chat_pdf(
            st.session_state.chat_history,
            answer_style,
            st.session_state.processed_file_names
        )

        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_data_sidebar,
            file_name="Answers.pdf",
            mime="application/pdf"
        )

        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.answer_cache = {}
            st.rerun()


# -----------------------------
# 17. PROCESS UPLOADED NOTES
# -----------------------------
if process_button:
    if not uploaded_files:
        st.warning("Please upload at least one notes file.")
    else:
        with st.spinner("Extracting text from uploaded notes..."):
            extracted_text = extract_text_from_uploaded_files(uploaded_files)

        if not extracted_text.strip():
            st.error("No text could be extracted from the uploaded files.")
        else:
            with st.spinner("Splitting text into chunks..."):
                chunks = create_chunks(extracted_text)

            with st.spinner("Processing the file..."):
                vector_store = create_vector_store(chunks)

            st.session_state.vector_store = vector_store
            st.session_state.chunks = chunks
            st.session_state.processed_file_names = [file.name for file in uploaded_files]

            st.session_state.answer_cache = {}
            st.session_state.chat_history = []

            st.success("Notes processed successfully!")
            st.info("Old chat history and answer cache cleared because new notes were processed.")


# -----------------------------
# 18. MAIN CONTENT LAYOUT
# -----------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("## 💬 Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: Explain activation functions"
    )

    ask_button = st.button("🔍 Ask Question")

with right_col:
    st.markdown(
        """
        <div class='glass-card'>
            <div style="font-family:'Playfair Display',serif; font-size:18px; color:#f5b754; margin-bottom:10px;">📌 Cheat sheet</div>
            <ul>
                <li>Upload PDF, DOCX, or TXT notes</li>
                <li>Ask questions straight from your notes</li>
                <li>Every Q&A is saved to history</li>
                <li>Export the full session as a PDF</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# 19. QUESTION ANSWERING SECTION
# -----------------------------
def render_answer(answer_text):
    """Render the answer as either a normal answer-card or a distinct
    'not found in notes' empty-state card, depending on content."""

    if answer_text.strip().lower() == NO_INFO_TEXT.lower():
        st.markdown(
            """
            <div class='noinfo-card'>
                <h2>🔍 Not in your notes</h2>
                <p>This question doesn't seem to be covered in the notes you uploaded.
                Try rephrasing it, or upload a file that covers this topic.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        safe_answer = clean_text_for_html(answer_text)
        st.markdown(
            f"""
            <div class='answer-card'>
                <h2>✅ Answer</h2>
                <div class="answer-text">{safe_answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


if ask_button:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process notes first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        cache_key = make_cache_key(question, answer_style)

        if cache_key in st.session_state.answer_cache:
            cached_data = st.session_state.answer_cache[cache_key]
            answer = cached_data["answer"]
            docs = cached_data["sources"]

        else:
            with st.spinner("Checking document relevance..."):
                docs, is_related, best_score = retrieve_relevant_docs(
                    st.session_state.vector_store,
                    question,
                    k=6,
                    max_score=1.35
                )

            if not is_related:
                answer = NO_INFO_TEXT
                docs = []
            else:
                context = "\n\n".join([doc.page_content for doc in docs])

                with st.spinner("Generating answer from uploaded notes..."):
                    answer = ask_groq(
                        question=question,
                        context=context,
                        answer_style=answer_style
                    )

            st.session_state.answer_cache[cache_key] = {
                "answer": answer,
                "sources": docs
            }

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "sources": docs,
                "answer_style": answer_style
            }
        )

        render_answer(answer)


# -----------------------------
# 20. SHOW CHAT HISTORY
# -----------------------------
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## 🕘 Chat History")

    unique_chat_history = get_unique_chat_history(st.session_state.chat_history)

    for i, chat in enumerate(reversed(unique_chat_history), start=1):
        with st.expander(f"Q{i}: {chat['question']}"):
            safe_q = clean_text_for_html(chat["question"])

            if chat["answer"].strip().lower() == NO_INFO_TEXT.lower():
                st.markdown(
                    f"""
                    <div class="note-card">
                        <div class="note-q">Q: {safe_q}</div>
                        <div class="note-a" style="border-left-color:#93a1b8; color:#93a1b8 !important;">
                            🔍 Not found in the uploaded notes.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                safe_a = clean_text_for_html(chat["answer"])
                st.markdown(
                    f"""
                    <div class="note-card">
                        <div class="note-q">Q: {safe_q}</div>
                        <div class="note-a">{safe_a}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("---")

    pdf_data = create_chat_pdf(
        st.session_state.chat_history,
        answer_style,
        st.session_state.processed_file_names
    )

    st.download_button(
        label="⬇️ Download PDF",
        data=pdf_data,
        file_name="Answers.pdf",
        mime="application/pdf"
    )

else:
    st.markdown(
        """
        <div class='start-card'>
            <h3>👋 Start Here</h3>
            <p>Upload your notes in the sidebar, hit <b>Process Notes</b>, then ask away.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

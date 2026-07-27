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
    page_title="College Notes RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# -----------------------------
# 2. CUSTOM UI CSS
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image:
            linear-gradient(rgba(2, 6, 23, 0.78), rgba(2, 6, 23, 0.84)),
            url("https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 8px;
        letter-spacing: 1.5px;
        text-shadow: 0px 3px 18px rgba(59, 130, 246, 0.9);
    }

    .sub-title {
        text-align: center;
        font-size: 17px;
        color: #dbeafe;
        margin-bottom: 35px;
        font-weight: 500;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    p, li, label, span, div {
        color: #e5e7eb !important;
    }

    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.94) !important;
        border-right: 1px solid rgba(96, 165, 250, 0.45);
        backdrop-filter: blur(14px);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #ffffff !important;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.90) !important;
        border: 2px dashed #60a5fa !important;
        border-radius: 16px !important;
        padding: 12px !important;
    }

    div[data-testid="stFileUploader"] * {
        color: #ffffff !important;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.95) !important;
        border: 1px dashed #93c5fd !important;
        border-radius: 14px !important;
    }

    div[data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }

    div[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #2563eb, #06b6d4) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }

    div[data-testid="stFileUploaderDropzone"] button * {
        color: white !important;
    }

    div[data-testid="stFileUploaderFile"] {
        background: rgba(2, 6, 23, 0.98) !important;
        border: 1px solid #60a5fa !important;
        border-radius: 12px !important;
        padding: 8px !important;
    }

    div[data-testid="stFileUploaderFile"] * {
        color: #ffffff !important;
    }

    div[data-testid="stFileUploaderFileName"] {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    div[data-testid="stFileUploaderFileSize"] {
        color: #bfdbfe !important;
    }

    div[data-testid="stFileUploaderFile"] button {
        background: #2563eb !important;
        color: white !important;
        border-radius: 50% !important;
    }

    div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.96) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid #60a5fa !important;
    }

    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }

    div[data-baseweb="popover"] * {
        background-color: #0f172a !important;
        color: #ffffff !important;
    }

    input {
        background: rgba(15, 23, 42, 0.94) !important;
        color: #ffffff !important;
        border: 1px solid #60a5fa !important;
        border-radius: 12px !important;
    }

    input::placeholder {
        color: #cbd5e1 !important;
    }

    textarea {
        background: rgba(15, 23, 42, 0.94) !important;
        color: #ffffff !important;
        border: 1px solid #60a5fa !important;
        border-radius: 12px !important;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 800;
        border: none;
        background: linear-gradient(135deg, #2563eb, #06b6d4);
        color: white !important;
        padding: 11px 22px;
        transition: 0.3s;
        box-shadow: 0px 6px 18px rgba(37, 99, 235, 0.45);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #0891b2);
        color: white !important;
        transform: translateY(-2px);
    }

    .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 800;
        background: linear-gradient(135deg, #16a34a, #22c55e);
        color: white !important;
        border: none;
        padding: 11px 22px;
        box-shadow: 0px 6px 18px rgba(34, 197, 94, 0.35);
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #15803d, #16a34a);
        color: white !important;
    }

    .glass-card {
        background: rgba(15, 23, 42, 0.88);
        color: #ffffff;
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0px 8px 30px rgba(2, 6, 23, 0.45);
        border: 1px solid rgba(96, 165, 250, 0.45);
        margin-top: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }

    .glass-card * {
        color: #ffffff !important;
    }

    .answer-card {
        background: rgba(15, 23, 42, 0.90);
        color: #ffffff;
        padding: 24px;
        border-radius: 18px;
        border-left: 6px solid #38bdf8;
        box-shadow: 0px 8px 30px rgba(2, 6, 23, 0.45);
        margin-top: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        line-height: 1.7;
    }

    .answer-card * {
        color: #ffffff !important;
    }

    .answer-text {
        font-size: 17px;
        line-height: 1.8;
        color: #ffffff !important;
    }

    .answer-text b {
        color: #ffffff !important;
        font-weight: 900 !important;
        font-size: 18px;
    }

    .start-card {
        background: rgba(15, 23, 42, 0.90);
        color: #ffffff;
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0px 8px 30px rgba(2, 6, 23, 0.45);
        border: 1px solid rgba(96, 165, 250, 0.45);
        margin-top: 25px;
        backdrop-filter: blur(12px);
    }

    .start-card * {
        color: #ffffff !important;
    }

    div[data-testid="column"] * {
        color: #ffffff !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.92) !important;
        border: 1px solid rgba(96, 165, 250, 0.45) !important;
        border-radius: 14px !important;
    }

    div[data-testid="stExpander"] * {
        color: #ffffff !important;
    }

    .streamlit-expanderHeader {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    .stAlert {
        background: rgba(15, 23, 42, 0.92) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(96, 165, 250, 0.35) !important;
    }

    .stAlert * {
        color: #ffffff !important;
    }

    hr {
        border-color: rgba(96, 165, 250, 0.35) !important;
    }

    .stCaptionContainer, .stCaptionContainer * {
        color: #cbd5e1 !important;
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
def ask_groq(question, context, answer_style):
    if not GROQ_API_KEY:
        return "Groq API key not found. Please add GROQ_API_KEY."

    client = Groq(api_key=GROQ_API_KEY)

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
     No info
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

    story.append(Paragraph("College Notes RAG Chatbot", title_style))
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
    "<div class='main-title'>📚 College Notes RAG Chatbot</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Upload notes, ask questions, get answers, and download the full Q&A as a PDF.</div>",
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
    st.markdown("## 📊 Status")

    if st.session_state.vector_store is not None:
        st.success("Notes are ready.")
        if st.session_state.processed_file_names:
            st.markdown("**Processed Files:**")
            for file_name in st.session_state.processed_file_names:
                st.write(f"📄 {file_name}")
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
    st.markdown("## 💬 Ask Question from Your Notes")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: Explain activation functions"
    )

    ask_button = st.button("🔍 Ask Question")

with right_col:
    st.markdown(
        """
        <div class='glass-card'>
            <ul>
                <li>Upload PDF, DOCX, TXT notes</li>
                <li>Ask questions from notes</li>
                <li>Saves Q&A history</li>
                <li>Downloads full Q&A as PDF</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# 19. QUESTION ANSWERING SECTION
# -----------------------------
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
                answer = "No info"
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

        safe_answer = clean_text_for_html(answer)

        st.markdown(
            f"""
            <div class='answer-card'>
                <h2>✅ Answer</h2>
                <div class="answer-text">{safe_answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------
# 20. SHOW CHAT HISTORY
# -----------------------------
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## 🕘 Chat History")

    unique_chat_history = get_unique_chat_history(st.session_state.chat_history)

    for i, chat in enumerate(reversed(unique_chat_history), start=1):
        with st.expander(f"Q{i}: {chat['question']}"):
            st.markdown("Question:")
            st.write(chat["question"])

            st.markdown("Answer:")
            st.write(chat["answer"])

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
        </div>
        """,
        unsafe_allow_html=True
    )

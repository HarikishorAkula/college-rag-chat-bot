import os, re
from io import BytesIO
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from guardrails import check_input, sanitize_output, check_rate_limit

st.set_page_config(page_title="College Notes RAG Chatbot", page_icon="📚", layout="wide")
NO_INFO_TEXT = "No info"

# ---------- CSS: violet "night library" theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');
.stApp{background:linear-gradient(rgba(15,4,32,.82),rgba(10,3,28,.90)),
    url("https://images.unsplash.com/photo-1481627834876-b7833e8f5570?auto=format&fit=crop&w=1920&q=80");
    background-size:cover;background-position:center;background-attachment:fixed;
    color:#f1e9ff;font-family:'Inter',sans-serif}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2.3rem;padding-bottom:3rem}
h1,h2,h3,h4,h5,h6,p,li,label,span,div{color:#f1e9ff !important;font-family:'Inter',sans-serif}
.main-title{text-align:center;font-family:'Poppins',sans-serif;font-size:44px;font-weight:800;
    background:linear-gradient(90deg,#c084fc,#f0abfc);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;margin-bottom:4px}
.sub-title{text-align:center;font-size:15px;color:#d8b4fe !important;margin-bottom:28px}
section[data-testid="stSidebar"]{background:rgba(15,4,32,.95) !important;
    border-right:1px solid rgba(192,132,252,.3)}
div[data-testid="stFileUploader"],div[data-testid="stFileUploaderDropzone"]{
    background:rgba(30,10,55,.7) !important;border:1px dashed #c084fc !important;border-radius:12px !important}
div[data-testid="stFileUploaderDropzone"] button{background:linear-gradient(135deg,#a855f7,#ec4899) !important;
    color:#fff !important;border-radius:8px !important;border:none !important}
div[data-testid="stFileUploaderFile"]{background:rgba(10,3,28,.95) !important;
    border:1px solid #c084fc !important;border-radius:10px !important}
div[data-baseweb="select"]>div,input,textarea{background:rgba(30,10,55,.75) !important;color:#f1e9ff !important;
    border:1px solid #c084fc !important;border-radius:10px !important}
div[data-baseweb="popover"] *{background-color:#1e0a37 !important;color:#f1e9ff !important}
.stButton>button{border-radius:10px;font-weight:700;font-family:'Poppins',sans-serif;border:none;
    background:linear-gradient(135deg,#a855f7,#ec4899);color:#fff !important;padding:10px 22px;
    box-shadow:0 6px 18px rgba(168,85,247,.4);transition:.2s}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 22px rgba(236,72,153,.5)}
.stDownloadButton>button{border-radius:10px;font-weight:700;font-family:'Poppins',sans-serif;border:none;
    background:linear-gradient(135deg,#22d3ee,#818cf8);color:#0a041c !important;padding:10px 22px;
    box-shadow:0 6px 18px rgba(34,211,238,.35)}
.glass-card,.answer-card,.start-card{background:rgba(30,10,55,.65);backdrop-filter:blur(10px);
    border-radius:16px;padding:22px;margin:18px 0;border:1px solid rgba(192,132,252,.3)}
.answer-card{border-left:5px solid #ec4899}
.answer-card h2,.start-card h3{font-family:'Poppins',sans-serif;color:#f0abfc !important}
.noinfo-card{background:rgba(30,10,55,.5);border-left:5px solid #94a3b8;border-radius:16px;
    padding:22px;margin:18px 0}
.answer-text{font-size:16px;line-height:1.8}
.answer-text b{font-size:17px;font-weight:800;color:#f0abfc !important}
.note-card{background:rgba(30,10,55,.55);border-left:3px solid #ec4899;border-radius:10px;
    padding:14px 16px;margin-bottom:4px}
.note-q{color:#f0abfc !important;font-weight:700;margin-bottom:6px;font-family:'Poppins',sans-serif}
.status-badge{display:inline-block;font-size:12px;padding:4px 10px;border-radius:6px;
    border:1px solid #c084fc;color:#e9d5ff !important;margin:2px 4px 2px 0}
div[data-testid="stExpander"]{background:rgba(30,10,55,.7) !important;
    border:1px solid rgba(192,132,252,.3) !important;border-radius:12px !important}
.stAlert{background:rgba(30,10,55,.75) !important;border-radius:10px !important;
    border:1px solid rgba(192,132,252,.3) !important}
hr{border-color:rgba(192,132,252,.3) !important}
</style>
""", unsafe_allow_html=True)

# ---------- API ----------
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

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------- TEXT EXTRACTION ----------
def extract_text(f):
    name = f.name.lower()
    try:
        if name.endswith(".pdf"):
            parts = [f"\n\n--- Source: {f.name}, Page: {i} ---\n{p}"
                     for i, page in enumerate(PdfReader(f).pages, start=1) if (p := page.extract_text())]
            return "".join(parts)
        if name.endswith(".docx"):
            text = "\n".join(p.text for p in Document(f).paragraphs if p.text.strip())
            return f"\n\n--- Source: {f.name} ---\n{text}"
        if name.endswith(".txt"):
            return f"\n\n--- Source: {f.name} ---\n{f.read().decode('utf-8', errors='ignore')}"
        st.warning(f"Unsupported file type: {f.name}")
        return ""
    except Exception as e:
        st.error(f"Could not read {f.name}: {e}")
        return ""

def create_chunks(text):
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
    ).split_text(text)

def retrieve_relevant_docs(vector_store, question, k=6, max_score=1.35):
    results = vector_store.similarity_search_with_score(question, k=k)
    if not results or results[0][1] > max_score:
        return [], False
    return [doc for doc, _ in results], True

# ---------- LLM ----------
# NOTE: Groq periodically retires/renames models. If you see a "model not found"
# or "decommissioned" error, update GROQ_MODEL below to a currently supported
# model name from https://console.groq.com/docs/models
GROQ_MODEL = "llama-3.3-70b-versatile"

def ask_groq(question, context, answer_style):
    if not GROQ_API_KEY:
        return "Groq API key not found. Please add GROQ_API_KEY to your .env file or Streamlit secrets."

    prompt = f"""
You are a College Notes Assistant.

Priority rules:
1. First preference is always the uploaded document context.
2. If the answer is clearly available in the context, answer only using it, no outside info.
3. If related but not fully present, start with:
   This exact answer was not found in the uploaded notes, but here is a general explanation:
   then give a simple general explanation.
4. If unrelated to the uploaded document, reply only: {NO_INFO_TEXT}
5. Do not use markdown symbols like **, *, ###.
6. Use short clear headings like Definition:, Role:, Example:
7. Use simple student-friendly language.
8. Format the answer according to this style: {answer_style}

Uploaded Document Context:
{context}

Student Question:
{question}

Answer:
"""
    try:
        client = get_groq_client(GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return (
            "Error while contacting Groq API. "
            f"Details: {e}\n\n"
            "If this mentions a decommissioned/unknown model, update GROQ_MODEL "
            "in the code to a current model from https://console.groq.com/docs/models."
        )

# ---------- HELPERS ----------
def make_cache_key(q, style):
    return f"{q.strip().lower()}__{style.strip().lower()}"

def clean_text(text, for_pdf=False):
    if text is None:
        return ""
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", str(text))
    text = text.replace("* ", "").replace("*", "")
    text = escape(text).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    if for_pdf:
        return text.replace("\n", "<br/>")
    lines = [f"<b>{l.strip()}</b>" if l.strip().endswith(":") and len(l.strip()) <= 60 else l
             for l in text.split("\n")]
    return "<br>".join(lines)

def get_unique_chat_history(chat_history):
    seen, unique = set(), []
    for chat in chat_history:
        q = chat.get("question", "").strip().lower()
        if q and q not in seen:
            unique.append(chat)
            seen.add(q)
    return unique

def create_chat_pdf(chat_history):
    unique = get_unique_chat_history(chat_history)
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"], alignment=TA_CENTER, fontSize=20,
        textColor=colors.HexColor("#6d28d9"), spaceAfter=18)
    sub_style = ParagraphStyle("S", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10,
        textColor=colors.HexColor("#64748b"), spaceAfter=25)
    q_style = ParagraphStyle("Q", parent=styles["Heading3"], fontSize=12,
        textColor=colors.HexColor("#a21caf"), spaceBefore=12, spaceAfter=8)
    meta_style = ParagraphStyle("M", parent=styles["Normal"], fontSize=9, spaceAfter=10)

    story = [Paragraph("College Notes RAG Chatbot", title_style),
             Paragraph("Questions and Answers Report", sub_style),
             Paragraph(f"<b>Total Questions:</b> {len(unique)}", meta_style), Spacer(1, 12)]
    for i, chat in enumerate(unique, start=1):
        story.append(Paragraph(f"Question {i}: {clean_text(chat.get('question',''), True)}", q_style))
        story.append(Paragraph(f"<b>Answer:</b><br/>{clean_text(chat.get('answer',''), True)}"))
        story.append(Spacer(1, 12))
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ---------- SESSION STATE ----------
for k, d in [("vector_store", None), ("chat_history", []), ("answer_cache", {}), ("processed_file_names", [])]:
    st.session_state.setdefault(k, d)

# ---------- HEADER ----------
st.markdown("<div class='main-title'>📚 College Notes RAG Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload notes, ask questions, get answers, and download the full Q&A as a PDF.</div>",
            unsafe_allow_html=True)

if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY not found. Add it to a `.env` file (GROQ_API_KEY=your_key) "
               "or to `.streamlit/secrets.toml` before asking questions.")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 📤 Upload Notes")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT notes",
                                       type=["pdf", "docx", "txt"], accept_multiple_files=True)
    answer_style = st.selectbox("Choose answer style",
        ["Simple explanation", "Detailed explanation", "Exam point of view", "Short notes", "Important points"])
    process_button = st.button("🚀 Process Notes")

    st.markdown("---")
    st.markdown("## 📊 Status")
    if st.session_state.vector_store is not None:
        st.success("Notes are ready.")
        badges = "".join(f"<span class='status-badge'>📄 {n}</span>" for n in st.session_state.processed_file_names)
        st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
    else:
        st.info("Upload and process notes first.")

    st.markdown("---")
    if st.session_state.chat_history:
        st.download_button("⬇️ Download PDF", data=create_chat_pdf(st.session_state.chat_history),
                            file_name="Answers.pdf", mime="application/pdf")
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history, st.session_state.answer_cache = [], {}
            st.rerun()

# ---------- PROCESS NOTES ----------
if process_button:
    if not uploaded_files:
        st.warning("Please upload at least one notes file.")
    else:
        with st.spinner("Extracting text..."):
            extracted_text = "".join(extract_text(f) for f in uploaded_files)
        if not extracted_text.strip():
            st.error("No text could be extracted from the uploaded files.")
        else:
            try:
                with st.spinner("Splitting text into chunks..."):
                    chunks = create_chunks(extracted_text)
                with st.spinner("Building vector store..."):
                    st.session_state.vector_store = FAISS.from_texts(chunks, embedding=load_embedding_model())
                st.session_state.processed_file_names = [f.name for f in uploaded_files]
                st.session_state.answer_cache, st.session_state.chat_history = {}, []
                st.success("Notes processed successfully!")
                st.info("Old chat history and answer cache cleared because new notes were processed.")
            except Exception as e:
                st.error(f"Failed to process notes: {e}")

# ---------- MAIN LAYOUT ----------
left_col, right_col = st.columns([2, 1])
with left_col:
    st.markdown("## 💬 Ask Question from Your Notes")
    question = st.text_input("Enter your question", placeholder="Example: Explain activation functions")
    ask_button = st.button("🔍 Ask Question")
with right_col:
    st.markdown("""<div class='glass-card'><ul>
        <li>Upload PDF, DOCX, TXT notes</li><li>Ask questions from notes</li>
        <li>Saves Q&A history</li><li>Downloads full Q&A as PDF</li></ul></div>""",
        unsafe_allow_html=True)

# ---------- ASK QUESTION ----------
def render_answer(answer_text):
    if answer_text.strip().lower() == NO_INFO_TEXT.lower():
        st.markdown("""<div class='noinfo-card'><h2>🔍 Not in your notes</h2>
            <p>This question doesn't seem to be covered in the notes you uploaded.
            Try rephrasing it, or upload a file that covers this topic.</p></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='answer-card'><h2>✅ Answer</h2>
            <div class="answer-text">{clean_text(answer_text)}</div></div>""", unsafe_allow_html=True)

if ask_button:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process notes first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        input_ok, input_msg = check_input(question)
        rate_ok, rate_msg = check_rate_limit(st.session_state)

        if not input_ok:
            st.warning(input_msg)
        elif not rate_ok:
            st.warning(rate_msg)
        else:
            cache_key = make_cache_key(question, answer_style)
            if cache_key in st.session_state.answer_cache:
                answer = st.session_state.answer_cache[cache_key]["answer"]
            else:
                try:
                    with st.spinner("Checking document relevance..."):
                        docs, is_related = retrieve_relevant_docs(st.session_state.vector_store, question)
                    if not is_related:
                        answer = NO_INFO_TEXT
                    else:
                        context = "\n\n".join(doc.page_content for doc in docs)
                        with st.spinner("Generating answer from uploaded notes..."):
                            answer = ask_groq(question, context, answer_style)
                    answer = sanitize_output(answer)
                except Exception as e:
                    answer = f"Something went wrong while retrieving or answering: {e}"
                st.session_state.answer_cache[cache_key] = {"answer": answer}

        st.session_state.chat_history.append({"question": question, "answer": answer, "answer_style": answer_style})
        render_answer(answer)

# ---------- CHAT HISTORY ----------
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## 🕘 Chat History")
    for i, chat in enumerate(reversed(get_unique_chat_history(st.session_state.chat_history)), start=1):
        with st.expander(f"Q{i}: {chat['question']}"):
            is_noinfo = chat["answer"].strip().lower() == NO_INFO_TEXT.lower()
            answer_html = "🔍 Not found in the uploaded notes." if is_noinfo else clean_text(chat["answer"])
            st.markdown(f"""<div class="note-card"><div class="note-q">Q: {clean_text(chat['question'])}</div>
                <div>{answer_html}</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.download_button("⬇️ Download PDF", data=create_chat_pdf(st.session_state.chat_history),
                        file_name="Answers.pdf", mime="application/pdf")
else:
    st.markdown("<div class='start-card'><h3>👋 Start Here</h3></div>", unsafe_allow_html=True)

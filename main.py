# =============================================================================
#  Gen AI RAG Solution + ATS Resume Checker (main.py)
#  Compatible with: langchain 0.3.x | langchain-core 0.3.x | Python 3.14
#
#  Install:
#  pip install streamlit langchain==0.3.25 langchain-core langchain-community
#             langchain-text-splitters langchain-groq langchain-huggingface
#             sentence-transformers pypdf faiss-cpu beautifulsoup4 python-dotenv
# =============================================================================

import os
import re
import tempfile
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv

# ── Text Splitters ─────────────────────────────────────────────────────────────
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)

# ── Core types ─────────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ── Document loaders ───────────────────────────────────────────────────────────
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    CSVLoader,
)

# ── Embeddings ─────────────────────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings

# ── Vector store (FAISS — works on all Python versions) ───────────────────────
from langchain_community.vectorstores import FAISS
VECTOR_BACKEND = "faiss"

# ── LLM ────────────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq

load_dotenv()

# =============================================================================
#  CONFIGURATION
# =============================================================================
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL     = "llama3-8b-8192"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# =============================================================================
#  STEP 1 – DATA SOURCES
# =============================================================================

def load_pdf(path):
    return PyPDFLoader(path).load()

def load_text(path):
    return TextLoader(path, encoding="utf-8").load()

def load_csv(path):
    return CSVLoader(path).load()

def load_word(path):
    try:
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        return UnstructuredWordDocumentLoader(path).load()
    except Exception as e:
        st.warning(f"Word loader error: {e}")
        return []

def load_html_file(path):
    try:
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        return UnstructuredHTMLLoader(path).load()
    except Exception as e:
        st.warning(f"HTML loader error: {e}")
        return []

def load_url(url):
    return WebBaseLoader(url).load()

def load_raw_text(text, source="manual_input"):
    return [Document(page_content=text, metadata={"source": source})]

LOADER_MAP = {
    ".pdf":  load_pdf,
    ".txt":  load_text,
    ".csv":  load_csv,
    ".docx": load_word,
    ".html": load_html_file,
    ".htm":  load_html_file,
}

def load_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    loader_fn = LOADER_MAP.get(suffix)
    if loader_fn:
        return loader_fn(tmp_path)
    st.warning(f"Unsupported file type: {suffix}")
    return []

# =============================================================================
#  STEP 2 – DATA PREPROCESSING
# =============================================================================

def preprocess_documents(docs):
    cleaned = []
    for doc in docs:
        text = doc.page_content
        text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()
        if len(text) >= 30:
            doc.page_content = text
            cleaned.append(doc)
    return cleaned

# =============================================================================
#  STEP 3 – SPLITTING & CHUNKING
# =============================================================================

def split_documents(docs, strategy="recursive"):
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(docs)
    elif strategy == "character":
        splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator="\n"
        )
        return splitter.split_documents(docs)
    elif strategy == "token":
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-3.5-turbo", chunk_size=300, chunk_overlap=50
        )
        return splitter.split_documents(docs)
    elif strategy == "markdown":
        headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers, strip_headers=False
        )
        chunks = []
        for doc in docs:
            chunks.extend(md_splitter.split_text(doc.page_content))
        return chunks
    elif strategy == "html":
        headers = [("h1", "H1"), ("h2", "H2"), ("h3", "H3")]
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers)
        chunks = []
        for doc in docs:
            chunks.extend(html_splitter.split_text(doc.page_content))
        return chunks
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        ).split_documents(docs)

# =============================================================================
#  STEP 4 – EMBEDDINGS + VECTOR DB
# =============================================================================

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_vector_store(chunks, embeddings):
    if not chunks:
        st.error("No chunks to embed.")
        return None
    vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# =============================================================================
#  STEP 5 – LLM + RAG CHAIN
# =============================================================================

def get_llm(api_key):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=LLM_MODEL,
        temperature=0.3,
        max_tokens=1024,
    )

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant. Answer ONLY using the context below.\n"
     "If the answer is not in the context, say: "
     "'I don't have enough information to answer that.'\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_rag_chain(retriever, llm):
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda x: format_docs(retriever.invoke(x["question"]))
            )
        )
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

# =============================================================================
#  SIDEBAR
# =============================================================================

def render_sidebar():
    st.sidebar.title("⚙️ Configuration")

    # API key status
    if GROQ_API_KEY:
        st.sidebar.success("🔑 API Key loaded from .env")
    else:
        st.sidebar.error("⚠️ GROQ_API_KEY not found in .env")
    st.sidebar.divider()

    st.sidebar.subheader("📂 Step 1 – Data Sources")
    uploaded_files = st.sidebar.file_uploader(
        "Upload files (PDF, TXT, CSV, DOCX, HTML)",
        type=["pdf", "txt", "csv", "docx", "html", "htm"],
        accept_multiple_files=True,
    )
    web_url  = st.sidebar.text_input("Or paste a URL to scrape", placeholder="https://…")
    raw_text = st.sidebar.text_area("Or paste raw text", height=100)
    st.sidebar.divider()

    st.sidebar.subheader("✂️ Step 3 – Chunking Strategy")
    strategy = st.sidebar.selectbox(
        "Splitter", ["recursive", "character", "token", "markdown", "html"], index=0
    )

    build_btn = st.sidebar.button("🚀 Build Knowledge Base", use_container_width=True)

    return {
        "uploaded_files": uploaded_files,
        "web_url":        web_url,
        "raw_text":       raw_text,
        "strategy":       strategy,
        "build_btn":      build_btn,
    }

# =============================================================================
#  INGEST DATA (Steps 1-3)
# =============================================================================

def ingest_data(cfg):
    raw_docs = []
    for uf in cfg["uploaded_files"]:
        with st.spinner(f"Loading {uf.name}…"):
            raw_docs.extend(load_uploaded_file(uf))
    if cfg["web_url"].strip():
        with st.spinner(f"Scraping {cfg['web_url']}…"):
            try:
                raw_docs.extend(load_url(cfg["web_url"].strip()))
            except Exception as e:
                st.warning(f"URL load failed: {e}")
    if cfg["raw_text"].strip():
        raw_docs.extend(load_raw_text(cfg["raw_text"].strip()))
    if not raw_docs:
        return []
    with st.spinner("Step 2 – Preprocessing…"):
        cleaned = preprocess_documents(raw_docs)
    with st.spinner(f"Step 3 – Chunking ({cfg['strategy']})…"):
        chunks = split_documents(cleaned, strategy=cfg["strategy"])
    return chunks

# =============================================================================
#  ATS CHECKER HELPER
# =============================================================================

def run_ats_analysis(resume_text: str, job_desc: str, api_key: str) -> str:
    prompt = f"""You are an expert ATS (Applicant Tracking System) analyst and career coach.

Analyse the resume below against the job description and respond with EXACTLY this structure:

## ATS Match Score
Give a single percentage score (0-100%) based on keyword overlap, skills match, experience relevance, and formatting. Write it clearly like: **Score: 72%**

## ✅ Matching Keywords & Skills
List the keywords and skills from the job description that ARE present in the resume (bullet points).

## ❌ Missing Keywords & Skills
List important keywords and skills from the job description that are MISSING from the resume (bullet points).

## 📈 What to Improve
Give 5-7 specific, actionable improvements the candidate should make to their resume to better match this job. Be concrete — mention exact sections, wording, or skills to add.

## 💡 Overall Recommendation
A 2-3 sentence summary of the candidate's fit and their single most important action to take.

---
RESUME:
{resume_text[:4000]}

---
JOB DESCRIPTION:
{job_desc[:2000]}
"""
    llm = get_llm(api_key)
    result = llm.invoke(prompt)
    return result.content

# =============================================================================
#  MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Gen AI RAG + ATS Checker",
        page_icon="🤖",
        layout="wide",
    )
    st.title("🤖 Gen AI RAG Assistant + ATS Resume Checker")
    st.caption("Pipeline: Data Sources → Preprocessing → Chunking → Embeddings → FAISS → Groq/Llama3")

    # Session state
    defaults = {
        "retriever":       None,
        "rag_chain":       None,
        "chat_history":    [],
        "display_history": [],
        "kb_ready":        False,
        "chunk_count":     0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    api_key = GROQ_API_KEY
    cfg     = render_sidebar()

    # ── Build Knowledge Base ─────────────────────────────────────────────────
    if cfg["build_btn"]:
        if not api_key:
            st.sidebar.error("⚠️ No API key. Add GROQ_API_KEY to your .env file.")
        else:
            chunks = ingest_data(cfg)
            if not chunks:
                st.sidebar.warning("No data loaded. Add files, a URL, or paste text.")
            else:
                with st.spinner("Step 4 – Embedding & building FAISS store…"):
                    embeddings = get_embeddings()
                    retriever  = build_vector_store(chunks, embeddings)
                if retriever:
                    with st.spinner("Step 5 – Initialising AI engine…"):
                        llm   = get_llm(api_key)
                        chain = build_rag_chain(retriever, llm)
                    st.session_state.retriever       = retriever
                    st.session_state.rag_chain       = chain
                    st.session_state.kb_ready        = True
                    st.session_state.chunk_count     = len(chunks)
                    st.session_state.chat_history    = []
                    st.session_state.display_history = []
                    st.sidebar.success(f"✅ Ready! ({len(chunks)} chunks | FAISS)")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_chat, tab_ats = st.tabs(["💬 RAG Chat", "📊 ATS Resume Checker"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — RAG CHAT
    # ════════════════════════════════════════════════════════════════════════
    with tab_chat:
        c1, c2, c3 = st.columns(3)
        c1.metric("Knowledge Base", "✅ Ready" if st.session_state.kb_ready else "⏳ Not Built")
        c2.metric("Chunks Indexed", st.session_state.chunk_count)
        c3.metric("Vector Backend", "FAISS")
        st.divider()

        st.subheader("💬 Chat with your Knowledge Base")

        for msg in st.session_state.display_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input(
            "Ask a question about your documents…",
            disabled=not st.session_state.kb_ready,
        )

        if user_input:
            st.session_state.display_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer  = st.session_state.rag_chain.invoke({
                            "question":     user_input,
                            "chat_history": st.session_state.chat_history,
                        })
                        sources = st.session_state.retriever.invoke(user_input)
                    except Exception as e:
                        answer  = f"❌ Error: {e}"
                        sources = []
                st.markdown(answer)
                if sources:
                    with st.expander("📄 Source References"):
                        for i, src in enumerate(sources, 1):
                            label = src.metadata.get("source", f"chunk {i}")
                            st.markdown(f"**[{i}] {label}**")
                            st.caption(src.page_content[:300] + "…")

            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=answer))
            st.session_state.display_history.append({"role": "assistant", "content": answer})

        if st.session_state.display_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history    = []
                st.session_state.display_history = []
                st.rerun()

        with st.expander("🔍 Pipeline Architecture"):
            st.markdown("""
| Step | Component | Tool |
|------|-----------|------|
| 1 | Data Sources | PyPDFLoader, WebBaseLoader, TextLoader, CSVLoader |
| 2 | Preprocessing | Regex cleaning, whitespace normalization, empty-page filter |
| 3 | Chunking | RecursiveCharacterTextSplitter (+ Markdown / HTML / Token / Character) |
| 4 | Embeddings | `all-MiniLM-L6-v2` via HuggingFace (free) |
| 4 | Vector DB | FAISS (in-memory, Python 3.14 compatible) |
| 5 | LLM | Llama 3 8B via Groq (free) |
| 5 | Chain | LCEL — RunnablePassthrough + RunnableLambda |
| 5 | Memory | HumanMessage / AIMessage (in-session) |
| 6 | UI | Streamlit |
            """)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — ATS RESUME CHECKER
    # ════════════════════════════════════════════════════════════════════════
    with tab_ats:
        st.subheader("📊 ATS Resume Checker")
        st.caption("Upload your resume + paste a job description → get ATS match % and improvement tips")
        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 📄 Your Resume")
            resume_file = st.file_uploader(
                "Upload Resume (PDF or TXT)",
                type=["pdf", "txt"],
                key="ats_resume",
            )
            if resume_file:
                st.success(f"✅ Uploaded: {resume_file.name}")

        with col_right:
            st.markdown("#### 📋 Job Description")
            job_desc = st.text_area(
                "Paste the full job description here",
                height=280,
                key="ats_jd",
                placeholder="Paste the job description you are applying for…",
            )

        st.divider()
        run_ats = st.button("🔍 Analyse ATS Match", use_container_width=True, key="ats_btn")

        if run_ats:
            if not api_key:
                st.error("⚠️ No Groq API Key. Add GROQ_API_KEY to your .env file.")
            elif not resume_file:
                st.warning("⚠️ Please upload your resume (PDF or TXT).")
            elif not job_desc.strip():
                st.warning("⚠️ Please paste the job description.")
            else:
                # Extract resume text
                with st.spinner("Reading resume…"):
                    suffix = os.path.splitext(resume_file.name)[-1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(resume_file.getbuffer())
                        tmp_path = tmp.name
                    resume_docs = load_pdf(tmp_path) if suffix == ".pdf" else load_text(tmp_path)
                    resume_text = "\n".join(d.page_content for d in resume_docs)

                with st.spinner("🤖 Analysing with AI… (this takes ~10 seconds)"):
                    try:
                        ats_result = run_ats_analysis(resume_text, job_desc, api_key)
                    except Exception as e:
                        ats_result = f"❌ Error during analysis: {e}"

                st.divider()

                # Extract score and show big visual
                score_match = re.search(r'(\d{1,3})\s*%', ats_result)
                if score_match:
                    score = int(score_match.group(1))
                    score = min(score, 100)

                    if score >= 70:
                        color  = "#4caf50"
                        emoji  = "🟢"
                        grade  = "Strong Match"
                        bg     = "#0d2b0d"
                    elif score >= 40:
                        color  = "#ff9800"
                        emoji  = "🟡"
                        grade  = "Moderate Match"
                        bg     = "#2b1e0d"
                    else:
                        color  = "#f44336"
                        emoji  = "🔴"
                        grade  = "Weak Match"
                        bg     = "#2b0d0d"

                    # Score card
                    st.markdown(f"""
                    <div style="text-align:center; padding:32px; border-radius:16px;
                                background:{bg}; border:2px solid {color}; margin-bottom:24px;">
                        <div style="font-size:72px; line-height:1;">{emoji} {score}%</div>
                        <div style="font-size:22px; color:{color}; font-weight:700;
                                    margin-top:8px;">{grade}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Progress bar
                    st.progress(score / 100)
                    st.markdown("")

                # Full AI analysis
                st.markdown(ats_result)

                # Download button
                st.divider()
                st.download_button(
                    label="⬇️ Download ATS Report",
                    data=ats_result,
                    file_name="ats_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

# =============================================================================
#  ENTRY POINT  →  streamlit run main.py
# =============================================================================
if __name__ == "__main__":
    main()
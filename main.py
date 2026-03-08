# =============================================================================
#  🚀 INTELLIGENT RESUME ANALYZER — ADVANCED EDITION
#  Features: ATS Checker | Resume Rewriter | Cover Letter | Interview Prep
#            Skill Gap Roadmap | Multi-Job Matcher | RAG Chat | 📸 Camera Scan
#  UI: Clean White + Deep Indigo — Professional SaaS Theme
#  Stack: LangChain 0.3.x | Groq/Llama3 | FAISS | Streamlit | Tesseract OCR
# =============================================================================

import os, re, tempfile, warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

# ── macOS: point pytesseract to Homebrew Tesseract binary ──
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL     = "llama-3.3-70b-versatile"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# =============================================================================
#  CUSTOM CSS — CLEAN WHITE + DEEP INDIGO PROFESSIONAL SAAS THEME
# =============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #f8f9fc;
    font-family: 'Inter', sans-serif;
    color: #1e1b4b;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* ── HERO ── */
.hero-wrap {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    border-radius: 24px;
    padding: 56px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(67,56,202,0.25);
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(165,180,252,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(99,102,241,0.2) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 100px; padding: 6px 16px;
    font-size: 12px; font-weight: 600; color: #c7d2fe;
    letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(32px, 5vw, 52px);
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    letter-spacing: -1px;
    margin-bottom: 16px;
}
.hero-title span {
    background: linear-gradient(90deg, #a5b4fc, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 17px; color: #a5b4fc; font-weight: 400;
    line-height: 1.6; max-width: 560px;
}
.hero-pills {
    display: flex; flex-wrap: wrap; gap: 10px; margin-top: 28px;
}
.hero-pill {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 100px; padding: 6px 14px;
    font-size: 12px; color: #e0e7ff; font-weight: 500;
}

/* ── STAT CARDS ── */
.stat-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 28px; }
.stat-card {
    background: #ffffff;
    border: 1px solid #e0e7ff;
    border-radius: 16px; padding: 20px 24px;
    display: flex; align-items: center; gap: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(67,56,202,0.06);
    transition: all 0.2s;
}
.stat-card:hover { box-shadow: 0 4px 20px rgba(67,56,202,0.12); transform: translateY(-1px); }
.stat-icon {
    width: 44px; height: 44px; border-radius: 12px;
    background: #eef2ff;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}
.stat-info {}
.stat-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 20px; font-weight: 700; color: #1e1b4b;
}
.stat-label { font-size: 12px; color: #6b7280; font-weight: 500; margin-top: 2px; letter-spacing: 0.3px; }

/* ── SECTION HEADER ── */
.sec-header { margin-bottom: 20px; }
.sec-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 22px; font-weight: 700; color: #1e1b4b;
}
.sec-sub { font-size: 14px; color: #6b7280; margin-top: 4px; }

/* ── CARDS ── */
.result-card {
    background: #ffffff;
    border: 1px solid #e0e7ff;
    border-radius: 20px; padding: 32px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin: 20px 0;
}

/* ── CAMERA CARD ── */
.camera-info-card {
    background: linear-gradient(135deg, #f5f3ff, #eef2ff);
    border: 1px solid #c7d2fe;
    border-radius: 20px;
    padding: 28px 32px;
    margin: 16px 0 24px;
}
.camera-step {
    display: flex; align-items: flex-start; gap: 14px; margin-bottom: 16px;
}
.camera-step-num {
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, #4338ca, #6366f1);
    color: white; font-weight: 700; font-size: 13px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 2px;
}
.camera-step-text { font-size: 14px; color: #374151; line-height: 1.5; }
.camera-step-text strong { color: #4338ca; }

/* ── OCR PREVIEW ── */
.ocr-preview {
    background: #f8fafc;
    border: 1.5px solid #e0e7ff;
    border-radius: 16px;
    padding: 20px;
    font-family: 'Inter', monospace;
    font-size: 13px;
    color: #374151;
    line-height: 1.6;
    max-height: 280px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}
.ocr-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #dcfce7; border: 1px solid #bbf7d0;
    border-radius: 100px; padding: 4px 12px;
    font-size: 12px; font-weight: 600; color: #166534;
    margin-bottom: 10px;
}

/* ── SCORE CARD ── */
.score-wrap {
    background: linear-gradient(135deg, #1e1b4b, #4338ca);
    border-radius: 20px; padding: 40px;
    text-align: center; color: white; margin: 20px 0;
    box-shadow: 0 12px 40px rgba(67,56,202,0.3);
}
.score-label {
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #a5b4fc; margin-bottom: 12px;
}
.score-num {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 80px; font-weight: 800; line-height: 1; margin: 0;
}
.score-grade {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 18px; font-weight: 600;
    margin-top: 8px; letter-spacing: 1px;
}
.score-bar-wrap {
    background: rgba(255,255,255,0.15);
    border-radius: 100px; height: 8px;
    margin: 16px auto 0; max-width: 300px; overflow: hidden;
}
.score-bar-fill { height: 100%; border-radius: 100px; transition: width 1s ease; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border: 1px solid #e0e7ff !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 2px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 13px !important;
    color: #6b7280 !important; border-radius: 10px !important;
    padding: 8px 14px !important; transition: all 0.2s !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4338ca, #6366f1) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.35) !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: #f5f3ff !important; color: #4338ca !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #4338ca, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important; font-size: 14px !important;
    padding: 12px 24px !important; transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #ffffff !important;
    border: 1.5px solid #e0e7ff !important;
    border-radius: 12px !important;
    color: #1e1b4b !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder { color: #9ca3af !important; }

/* ── FILE UPLOADER ── */
.stFileUploader > div {
    background: #fafafa !important;
    border: 2px dashed #c7d2fe !important;
    border-radius: 16px !important;
    transition: all 0.2s !important;
}
.stFileUploader > div:hover {
    border-color: #6366f1 !important;
    background: #f5f3ff !important;
}

/* ── SELECTBOX ── */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid #e0e7ff !important;
    border-radius: 12px !important;
    color: #1e1b4b !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e0e7ff !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── ALERTS ── */
.stSuccess {
    background: #f0fdf4 !important; border: 1px solid #bbf7d0 !important;
    border-radius: 12px !important; color: #166534 !important;
}
.stError {
    background: #fef2f2 !important; border: 1px solid #fecaca !important;
    border-radius: 12px !important; color: #991b1b !important;
}
.stWarning {
    background: #fffbeb !important; border: 1px solid #fde68a !important;
    border-radius: 12px !important; color: #92400e !important;
}
.stInfo {
    background: #eff6ff !important; border: 1px solid #bfdbfe !important;
    border-radius: 12px !important; color: #1e40af !important;
}

/* ── PROGRESS ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #4338ca, #6366f1) !important;
    border-radius: 100px !important;
}
.stProgress > div > div {
    background: #e0e7ff !important; border-radius: 100px !important;
}

/* ── CHAT ── */
.stChatMessage {
    background: #ffffff !important;
    border: 1px solid #e0e7ff !important;
    border-radius: 16px !important; margin-bottom: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    color: #1e1b4b !important;
}
.stChatMessage p, .stChatMessage div,
.stChatMessage span, .stChatMessage li { color: #1e1b4b !important; }
[data-testid="stChatMessageContent"] { color: #1e1b4b !important; }
[data-testid="stChatMessageContent"] * { color: #1e1b4b !important; }
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1.5px solid #e0e7ff !important;
    border-radius: 12px !important; color: #1e1b4b !important;
}

/* ── SIDEBAR FILE UPLOADER FIX ── */
section[data-testid="stSidebar"] .stFileUploader > div {
    background: #f5f3ff !important;
    border: 2px dashed #c7d2fe !important;
    border-radius: 16px !important;
}
section[data-testid="stSidebar"] .stFileUploader p,
section[data-testid="stSidebar"] .stFileUploader span,
section[data-testid="stSidebar"] .stFileUploader div {
    color: #4338ca !important;
}
section[data-testid="stSidebar"] .stFileUploader button {
    background: #4338ca !important; color: white !important;
    border: none !important; border-radius: 8px !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e0e7ff;
    border-radius: 16px; padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #1e1b4b !important; font-family: 'Plus Jakarta Sans',sans-serif !important; font-weight: 700 !important; }

/* ── DOWNLOAD ── */
.stDownloadButton > button {
    background: #f5f3ff !important;
    border: 1.5px solid #c7d2fe !important;
    color: #4338ca !important; border-radius: 12px !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #4338ca, #6366f1) !important;
    color: white !important; border-color: transparent !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: #f8f9fc !important;
    border: 1px solid #e0e7ff !important;
    border-radius: 12px !important;
    color: #4338ca !important; font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important; height: 1px !important;
    background: #e0e7ff !important; margin: 24px 0 !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ── FEATURE CHIP ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.chip {
    background: #f5f3ff; border: 1px solid #c7d2fe;
    border-radius: 100px; padding: 6px 14px;
    font-size: 13px; font-weight: 500; color: #4338ca;
    cursor: pointer; transition: all 0.2s;
}
.chip:hover { background: #4338ca; color: white; }

/* ── SIDEBAR LOGO ── */
.sidebar-logo {
    background: linear-gradient(135deg, #1e1b4b, #4338ca);
    border-radius: 0 0 20px 20px;
    padding: 28px 20px 24px;
    margin-bottom: 4px;
    text-align: center;
}
.sidebar-brand {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 18px; font-weight: 800; color: white; letter-spacing: -0.5px;
}
.sidebar-tagline {
    font-size: 10px; color: #a5b4fc;
    letter-spacing: 2px; text-transform: uppercase;
    margin-top: 4px;
}

/* ── JOB CARD ── */
.job-col-header {
    background: linear-gradient(135deg, #f5f3ff, #eef2ff);
    border: 1px solid #e0e7ff; border-radius: 12px;
    padding: 12px 16px; margin-bottom: 12px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700; color: #4338ca; font-size: 15px;
}
</style>
"""

# =============================================================================
#  HELPERS
# =============================================================================

def load_pdf(path):   return PyPDFLoader(path).load()
def load_text(path):  return TextLoader(path, encoding="utf-8").load()

def extract_resume_text(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    docs = load_pdf(tmp_path) if suffix == ".pdf" else load_text(tmp_path)
    return "\n".join(d.page_content for d in docs)

def preprocess(text: str) -> str:
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

# =============================================================================
#  OCR — CAMERA / IMAGE → TEXT
# =============================================================================

def enhance_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Convert to grayscale and boost contrast for better OCR accuracy.
    Works on both colour photos and already-greyscale scans.
    """
    import PIL.ImageEnhance as IE, PIL.ImageFilter as IF
    img = pil_img.convert("L")                      # greyscale
    img = IE.Contrast(img).enhance(2.0)             # boost contrast
    img = IE.Sharpness(img).enhance(2.0)            # sharpen edges
    img = img.filter(IF.MedianFilter(size=3))       # reduce noise
    return img

def ocr_image(pil_img: Image.Image) -> str:
    """Run Tesseract OCR on a PIL image and return extracted text."""
    try:
        enhanced = enhance_image_for_ocr(pil_img)
        # PSM 6 = assume uniform block of text (good for resumes)
        config = "--psm 6 --oem 3"
        raw = pytesseract.image_to_string(enhanced, config=config)
        return preprocess(raw)
    except Exception as e:
        return f"OCR Error: {e}"

def ocr_quality_check(text: str) -> tuple[bool, str]:
    """Returns (is_good, message) based on extracted text quality."""
    word_count = len(text.split())
    if word_count < 30:
        return False, f"Only {word_count} words detected — try better lighting or a flatter surface."
    if word_count < 80:
        return True, f"⚠️ {word_count} words detected — quality may be low. Results might be imperfect."
    return True, f"✅ {word_count} words extracted — good quality scan!"

# =============================================================================
#  EMBEDDINGS / LLM
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_llm():
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL,
                    temperature=0.3, max_tokens=2048)

def llm_call(prompt: str) -> str:
    return get_llm().invoke(prompt).content

# =============================================================================
#  FEATURE FUNCTIONS
# =============================================================================

def ats_analysis(resume_text, job_desc):
    return llm_call(f"""You are a world-class ATS analyst and career coach.
Analyse the resume against the job description. Respond with EXACTLY this format:

## ATS MATCH SCORE
**Score: [NUMBER]%**
One sentence explaining the score.

## MATCHING KEYWORDS & SKILLS
List every keyword/skill from the JD that IS in the resume. One per line starting with ✅

## MISSING KEYWORDS & SKILLS
List every important keyword/skill from the JD that is MISSING. One per line starting with ❌

## SECTION SCORES
Rate each section out of 10:
- **Summary/Objective:** X/10 — reason
- **Work Experience:** X/10 — reason
- **Skills Section:** X/10 — reason
- **Education:** X/10 — reason
- **Formatting & ATS Compatibility:** X/10 — reason

## TOP 7 IMPROVEMENTS
Number each 1-7. Be very specific — mention exact changes to make.

## OVERALL VERDICT
2-3 sentences. Be direct and honest.

---
RESUME: {resume_text[:4000]}
JOB DESCRIPTION: {job_desc[:2000]}
""")

def rewrite_resume(resume_text, job_desc):
    return llm_call(f"""You are an expert resume writer.
Rewrite the resume to better match the job using STAR format, metrics, and action verbs.

## ✨ IMPROVED PROFESSIONAL SUMMARY
[Rewritten — 3-4 powerful sentences]

## 💼 STRENGTHENED EXPERIENCE BULLETS
For each role:
**[Job Title] at [Company]**
- BEFORE: [original]
  AFTER:  [improved with metrics]

## 🎯 OPTIMISED SKILLS SECTION
[Reorganised and expanded with missing keywords]

## 📋 ADDITIONAL RECOMMENDATIONS
3-5 specific structural improvements

---
RESUME: {resume_text[:4000]}
JOB DESCRIPTION: {job_desc[:2000]}
""")

def generate_cover_letter(resume_text, job_desc, tone):
    tone_map = {
        "Professional & Formal":  "formal, structured, traditional corporate",
        "Confident & Bold":       "confident, assertive, shows strong ambition",
        "Creative & Engaging":    "engaging storytelling, shows creativity and passion",
    }
    return llm_call(f"""Write a compelling cover letter. Tone: {tone_map[tone]}
Rules:
- 3-4 paragraphs, 250-350 words
- Opening: hook, mention specific role
- Body: connect 2-3 resume achievements to job requirements
- Closing: strong call to action
- Sound human, NOT AI-generated
- Never use: "I am writing to express my interest"

Format:
[Date]
Dear Hiring Manager,
[paragraphs]
Sincerely,
[Candidate Name]

---
RESUME: {resume_text[:3000]}
JOB DESCRIPTION: {job_desc[:2000]}
""")

def generate_interview_prep(resume_text, job_desc):
    return llm_call(f"""You are a senior interviewer at a top company.

## 🎯 TOP 5 TECHNICAL QUESTIONS
**Q[N]: [Question]**
Why asked: [reason]
Strong answer approach: [using candidate background]

## 🤝 TOP 5 BEHAVIOURAL QUESTIONS
**Q[N]: [Question]**
What tested: [reason]
STAR framework: [approach based on resume]

## 🧠 TOP 3 SITUATIONAL QUESTIONS
**Q[N]: [Question]**
How to approach: [strategy]

## ❓ 5 SMART QUESTIONS TO ASK INTERVIEWER

## ✅ QUICK PREP CHECKLIST
- [ ] Research the company
- [ ] Prepare "Tell me about yourself" (2 min)
- [ ] Have 3 achievement stories ready
- [ ] Know salary expectations
- [ ] Prepare your own questions

---
RESUME: {resume_text[:3000]}
JOB DESCRIPTION: {job_desc[:2000]}
""")

def generate_skill_roadmap(resume_text, job_desc):
    return llm_call(f"""You are a senior tech career advisor.

## 📊 SKILL GAP ANALYSIS
### ✅ Skills You Already Have
[skill — level: Beginner/Intermediate/Advanced]

### ❌ Critical Missing Skills
[skill — why critical]

### ⚠️ Nice to Have
[skill — priority]

## 🗺️ 90-DAY LEARNING ROADMAP
### Week 1-2: Quick Wins
- Skill: [name]
  Resource: [exact name + type]
  Time: [hours]
  Project: [mini project]

### Week 3-6: Core Skills [same format]
### Week 7-10: Advanced [same format]
### Week 11-12: Portfolio Projects
- Project 1: [name + skills demonstrated]
- Project 2: [name + skills demonstrated]

## ⏱️ JOB-READY TIMELINE
## 🆓 TOP 3 FREE PLATFORMS

---
RESUME: {resume_text[:3000]}
JOB DESCRIPTION: {job_desc[:2000]}
""")

def multi_job_match(resume_text, jobs):
    jobs_text = "".join(
        f"\nJOB {i} — {j['title']}:\n{j['desc'][:800]}\n"
        for i, j in enumerate(jobs, 1)
    )
    return llm_call(f"""You are an expert career advisor.

## 🏆 RANKING (Best Match First)
Rank jobs best to worst fit.

## 📊 DETAILED COMPARISON
For each job:
**Job [N]: [Title]**
- Match Score: X%
- Strongest Points: [2-3]
- Weakest Points: [2-3]
- Effort to compete: Low / Medium / High

## 🎯 RECOMMENDATION
Which to apply first and why. Be specific.

## 📋 TOP 3 RESUME TWEAKS PER JOB

---
RESUME: {resume_text[:3000]}
{jobs_text}
""")

def clean_ocr_with_llm(raw_ocr_text: str) -> str:
    """Use LLM to clean up and structure raw OCR text into a proper resume."""
    return llm_call(f"""You are an expert resume formatter.
The text below was extracted from a physical resume photo using OCR.
It may contain errors, garbled words, or broken formatting.

Your task:
1. Fix OCR errors (wrong characters, broken words)
2. Reconstruct proper resume sections (Summary, Experience, Skills, Education, etc.)
3. Clean up spacing and punctuation
4. Return ONLY the cleaned resume text — no commentary

RAW OCR TEXT:
{raw_ocr_text[:4000]}
""")

# =============================================================================
#  RAG CHAT
# =============================================================================

def build_rag(text: str):
    docs = [Document(page_content=preprocess(text), metadata={"source": "resume"})]
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, get_embeddings())
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent career assistant. Answer ONLY from this context:\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda x: "\n\n".join(d.page_content for d in retriever.invoke(x["question"]))
            )
        ) | prompt | get_llm() | StrOutputParser()
    )
    return retriever, chain

# =============================================================================
#  UI HELPERS
# =============================================================================

def score_card_html(score: int):
    score = min(score, 100)
    if score >= 75:   color, grade, emoji = "#10b981", "Strong Match",   "🟢"
    elif score >= 50: color, grade, emoji = "#f59e0b", "Moderate Match", "🟡"
    else:             color, grade, emoji = "#ef4444", "Weak Match",     "🔴"
    st.markdown(f"""
    <div class="score-wrap">
        <div class="score-label">ATS MATCH SCORE</div>
        <div class="score-num">{score}%</div>
        <div class="score-grade">{emoji} {grade}</div>
        <div class="score-bar-wrap">
            <div class="score-bar-fill" style="width:{score}%;background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def check_ready() -> bool:
    if not GROQ_API_KEY:
        st.error("⚠️ Add GROQ_API_KEY to your .env file")
        return False
    if not st.session_state.resume_text:
        st.warning("⬅️ Upload your resume in the sidebar first")
        return False
    return True

def jd_input(key, height=200):
    return st.text_area("📋 Paste Job Description", height=height, key=key,
                        placeholder="Paste the full job description here…")

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-brand">⚡ IRA</div>
            <div class="sidebar-tagline">Intelligent Resume Analyzer</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔑 API Status")
        if GROQ_API_KEY:
            st.success("API Key Active")
        else:
            st.error("Add GROQ_API_KEY to .env")

        st.divider()

        st.markdown("#### 📂 Upload Resume")
        resume_file = st.file_uploader(
            "PDF or TXT", type=["pdf","txt"], label_visibility="collapsed"
        )
        if resume_file:
            st.success(f"✅ {resume_file.name}")

        st.divider()

        st.markdown("""
        <div style="padding:12px 4px;">
            <div style="font-size:11px;color:#9ca3af;font-weight:600;letter-spacing:1px;margin-bottom:10px;">POWERED BY</div>
            <div style="display:flex;flex-wrap:wrap;gap:6px;">
                <span style="background:#f5f3ff;border:1px solid #e0e7ff;border-radius:8px;padding:3px 10px;font-size:11px;color:#4338ca;font-weight:600;">LangChain</span>
                <span style="background:#f5f3ff;border:1px solid #e0e7ff;border-radius:8px;padding:3px 10px;font-size:11px;color:#4338ca;font-weight:600;">Groq</span>
                <span style="background:#f5f3ff;border:1px solid #e0e7ff;border-radius:8px;padding:3px 10px;font-size:11px;color:#4338ca;font-weight:600;">FAISS</span>
                <span style="background:#f5f3ff;border:1px solid #e0e7ff;border-radius:8px;padding:3px 10px;font-size:11px;color:#4338ca;font-weight:600;">Llama 3</span>
                <span style="background:#f5f3ff;border:1px solid #e0e7ff;border-radius:8px;padding:3px 10px;font-size:11px;color:#4338ca;font-weight:600;">Streamlit</span>
                <span style="background:#f5f3ff;border:1px solid #e0e7ff;border-radius:8px;padding:3px 10px;font-size:11px;color:#4338ca;font-weight:600;">Tesseract</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return resume_file

# =============================================================================
#  CAMERA SCAN TAB
# =============================================================================

def render_camera_tab():
    st.markdown(
        '<div class="sec-header">'
        '<div class="sec-title">📸 Scan Physical Resume</div>'
        '<div class="sec-sub">Take a photo or upload an image of your printed resume — AI extracts and cleans the text automatically</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # How-to card
    st.markdown("""
    <div class="camera-info-card">
        <div class="camera-step">
            <div class="camera-step-num">1</div>
            <div class="camera-step-text"><strong>Place your resume</strong> on a flat, well-lit surface — avoid shadows and glare</div>
        </div>
        <div class="camera-step">
            <div class="camera-step-num">2</div>
            <div class="camera-step-text"><strong>Take a photo</strong> using the camera below, or upload an existing image (JPG/PNG)</div>
        </div>
        <div class="camera-step">
            <div class="camera-step-num">3</div>
            <div class="camera-step-text"><strong>Run OCR + AI Clean</strong> — Tesseract extracts text, then Llama 3 fixes errors & formats it</div>
        </div>
        <div class="camera-step">
            <div class="camera-step-num">4</div>
            <div class="camera-step-text"><strong>Use as your resume</strong> — one click loads it into all 7 features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input method toggle
    input_method = st.radio(
        "Input method",
        ["📷 Use Camera", "🖼️ Upload Image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    img = None

    if input_method == "📷 Use Camera":
        st.markdown("**📷 Point your camera at the resume:**")
        camera_img = st.camera_input(
            "Take a photo of your resume",
            label_visibility="collapsed",
            key="camera_snap",
        )
        if camera_img:
            img = Image.open(camera_img)

    else:
        uploaded_img = st.file_uploader(
            "Upload resume image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="img_upload",
        )
        if uploaded_img:
            img = Image.open(uploaded_img)

    if img:
        st.divider()

        # Show preview
        col_prev, col_info = st.columns([1, 1])
        with col_prev:
            st.markdown("**🖼️ Captured Image**")
            st.image(img, use_container_width=True)
        with col_info:
            st.markdown("**⚙️ OCR Options**")
            auto_clean = st.toggle(
                "✨ AI-clean extracted text with Llama 3",
                value=True,
                help="Fixes OCR errors, restructures sections using the LLM",
            )
            st.caption(f"Image size: {img.width} × {img.height} px")
            st.caption("Higher resolution = better accuracy")
            st.info("💡 **Tip:** Ensure text is horizontal and fills most of the frame for best results.")

        st.divider()

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            run_ocr = st.button("🔍 Extract Text (OCR)", use_container_width=True, key="btn_ocr")
        with col_btn2:
            if st.session_state.get("ocr_text"):
                load_btn = st.button("⚡ Use as My Resume", use_container_width=True, key="btn_load_ocr")
            else:
                st.button("⚡ Use as My Resume", use_container_width=True, key="btn_load_ocr_dis", disabled=True)

        # ── RUN OCR ──
        if run_ocr:
            with st.spinner("Running Tesseract OCR…"):
                raw_text = ocr_image(img)

            ok, quality_msg = ocr_quality_check(raw_text)
            st.markdown(f'<div class="ocr-badge">🔍 OCR Complete</div>', unsafe_allow_html=True)

            if not ok:
                st.error(f"⚠️ Low quality scan — {quality_msg}\nTry better lighting or upload a clearer photo.")
                st.session_state["ocr_text"] = ""
            else:
                if "⚠️" in quality_msg:
                    st.warning(quality_msg)
                else:
                    st.success(quality_msg)

                final_text = raw_text

                if auto_clean and GROQ_API_KEY:
                    with st.spinner("✨ AI is cleaning and formatting the extracted text…"):
                        final_text = clean_ocr_with_llm(raw_text)
                    st.success("✅ AI cleaning complete — text has been reconstructed")

                st.session_state["ocr_text"] = final_text

                # Show side by side if cleaned
                if auto_clean and GROQ_API_KEY:
                    tab_raw, tab_clean = st.tabs(["📄 Raw OCR", "✨ AI-Cleaned"])
                    with tab_raw:
                        st.markdown(f'<div class="ocr-preview">{raw_text}</div>', unsafe_allow_html=True)
                    with tab_clean:
                        st.markdown(f'<div class="ocr-preview">{final_text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("**📄 Extracted Text:**")
                    st.markdown(f'<div class="ocr-preview">{final_text}</div>', unsafe_allow_html=True)

                st.download_button(
                    "⬇️ Download Extracted Text",
                    final_text,
                    "scanned_resume.txt",
                    use_container_width=True,
                    key="dl_ocr",
                )

        # ── LOAD INTO APP ──
        if st.session_state.get("ocr_text") and st.session_state.get("btn_load_ocr"):
            text = st.session_state["ocr_text"]
            with st.spinner("Building AI knowledge base from scanned resume…"):
                _, chain = build_rag(text)
                st.session_state.resume_text       = text
                st.session_state.rag_chain         = chain
                st.session_state.rag_ready         = True
                st.session_state.chat_history      = []
                st.session_state.display_history   = []
            st.success("🎉 Scanned resume loaded! All 7 features are now ready to use.")
            st.balloons()

# =============================================================================
#  MAIN
# =============================================================================

def main():
    st.set_page_config(
        page_title="Intelligent Resume Analyzer",
        page_icon="⚡", layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Session state
    for k, v in {
        "resume_text": "", "rag_chain": None,
        "chat_history": [], "display_history": [],
        "rag_ready": False, "ocr_text": "",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    resume_file = render_sidebar()

    # Load + process resume (PDF/TXT upload)
    if resume_file:
        with st.spinner("Processing resume…"):
            new_text = preprocess(extract_resume_text(resume_file))
        if new_text != st.session_state.resume_text:
            st.session_state.resume_text = new_text
            with st.spinner("Building AI knowledge base…"):
                _, chain = build_rag(new_text)
                st.session_state.rag_chain       = chain
                st.session_state.rag_ready       = True
                st.session_state.chat_history    = []
                st.session_state.display_history = []

    # ── HERO ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">✦ AI-Powered Career Platform</div>
        <div class="hero-title">Intelligent Resume<br><span>Analyzer</span></div>
        <div class="hero-sub">
            Get your ATS score, rewrite your resume, generate cover letters,
            prepare for interviews, map your skill gaps — and now scan physical resumes with your camera.
        </div>
        <div class="hero-pills">
            <span class="hero-pill">📊 ATS Checker</span>
            <span class="hero-pill">✍️ Resume Rewriter</span>
            <span class="hero-pill">📝 Cover Letter</span>
            <span class="hero-pill">🎤 Interview Prep</span>
            <span class="hero-pill">🗺️ Skill Roadmap</span>
            <span class="hero-pill">🔀 Job Matcher</span>
            <span class="hero-pill">💬 RAG Chat</span>
            <span class="hero-pill">📸 Camera Scan</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── STATS ─────────────────────────────────────────────────────────────
    has_resume = bool(st.session_state.resume_text)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-icon">📄</div>
            <div class="stat-info">
                <div class="stat-value">{"Loaded ✓" if has_resume else "Not yet"}</div>
                <div class="stat-label">Resume Status</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🤖</div>
            <div class="stat-info">
                <div class="stat-value">{"Active ✓" if GROQ_API_KEY else "Offline"}</div>
                <div class="stat-label">AI Engine (Llama 3)</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🧠</div>
            <div class="stat-info">
                <div class="stat-value">{"Ready ✓" if st.session_state.rag_ready else "Standby"}</div>
                <div class="stat-label">RAG Knowledge Base</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 ATS Checker", "✍️ Resume Rewriter", "📝 Cover Letter",
        "🎤 Interview Prep", "🗺️ Skill Roadmap", "🔀 Job Matcher",
        "💬 RAG Chat", "📸 Camera Scan",
    ])

    # ── TAB 1: ATS ────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="sec-header"><div class="sec-title">📊 ATS Resume Checker</div><div class="sec-sub">Analyse your resume match score, missing keywords, and section-by-section scoring</div></div>', unsafe_allow_html=True)
        st.divider()
        jd = jd_input("jd_ats")
        if st.button("⚡ Run ATS Analysis", use_container_width=True, key="btn_ats"):
            if check_ready():
                if not jd.strip():
                    st.warning("Please paste a job description")
                else:
                    with st.spinner("Analysing with AI… (~15 seconds)"):
                        result = ats_analysis(st.session_state.resume_text, jd)
                    m = re.search(r'(\d{1,3})\s*%', result)
                    if m: score_card_html(int(m.group(1)))
                    st.divider()
                    st.markdown(result)
                    st.divider()
                    st.download_button("⬇️ Download ATS Report", result, "ats_report.txt", use_container_width=True)

    # ── TAB 2: REWRITER ───────────────────────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="sec-header"><div class="sec-title">✍️ AI Resume Rewriter</div><div class="sec-sub">STAR format bullets, quantified metrics, and keyword injection</div></div>', unsafe_allow_html=True)
        st.divider()
        jd = jd_input("jd_rewrite")
        if st.button("✨ Rewrite My Resume", use_container_width=True, key="btn_rewrite"):
            if check_ready():
                if not jd.strip():
                    st.warning("Please paste a job description")
                else:
                    with st.spinner("Rewriting your resume…"):
                        result = rewrite_resume(st.session_state.resume_text, jd)
                    st.markdown(result)
                    st.divider()
                    st.download_button("⬇️ Download Rewritten Resume", result, "rewritten_resume.txt", use_container_width=True)

    # ── TAB 3: COVER LETTER ───────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="sec-header"><div class="sec-title">📝 Cover Letter Generator</div><div class="sec-sub">Personalised, human-sounding cover letter tailored to the job</div></div>', unsafe_allow_html=True)
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            jd = jd_input("jd_cover")
        with col2:
            st.markdown("**Tone & Style**")
            tone = st.selectbox("Tone", [
                "Professional & Formal", "Confident & Bold", "Creative & Engaging"
            ], label_visibility="collapsed")
            st.caption("**Professional** — Corporate, structured")
            st.caption("**Confident** — Bold, assertive")
            st.caption("**Creative** — Story-driven, engaging")
        if st.button("📝 Generate Cover Letter", use_container_width=True, key="btn_cover"):
            if check_ready():
                if not jd.strip():
                    st.warning("Please paste a job description")
                else:
                    with st.spinner("Crafting your cover letter…"):
                        result = generate_cover_letter(st.session_state.resume_text, jd, tone)
                    st.markdown(result)
                    st.divider()
                    st.download_button("⬇️ Download Cover Letter", result, "cover_letter.txt", use_container_width=True)

    # ── TAB 4: INTERVIEW ──────────────────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="sec-header"><div class="sec-title">🎤 Interview Preparation</div><div class="sec-sub">Personalised questions with AI-suggested answers based on your resume</div></div>', unsafe_allow_html=True)
        st.divider()
        jd = jd_input("jd_interview")
        if st.button("🎤 Generate Interview Prep", use_container_width=True, key="btn_interview"):
            if check_ready():
                if not jd.strip():
                    st.warning("Please paste a job description")
                else:
                    with st.spinner("Preparing your interview guide…"):
                        result = generate_interview_prep(st.session_state.resume_text, jd)
                    st.markdown(result)
                    st.divider()
                    st.download_button("⬇️ Download Interview Guide", result, "interview_prep.txt", use_container_width=True)

    # ── TAB 5: ROADMAP ────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="sec-header"><div class="sec-title">🗺️ Skill Gap Roadmap</div><div class="sec-sub">90-day personalised learning roadmap with free resources and projects</div></div>', unsafe_allow_html=True)
        st.divider()
        jd = jd_input("jd_roadmap")
        if st.button("🗺️ Generate My Roadmap", use_container_width=True, key="btn_roadmap"):
            if check_ready():
                if not jd.strip():
                    st.warning("Please paste a job description")
                else:
                    with st.spinner("Building your personalised roadmap…"):
                        result = generate_skill_roadmap(st.session_state.resume_text, jd)
                    st.markdown(result)
                    st.divider()
                    st.download_button("⬇️ Download Roadmap", result, "skill_roadmap.txt", use_container_width=True)

    # ── TAB 6: JOB MATCHER ────────────────────────────────────────────────
    with tabs[5]:
        st.markdown('<div class="sec-header"><div class="sec-title">🔀 Multi-Job Matcher</div><div class="sec-sub">Compare your resume against 3 jobs — find your best fit instantly</div></div>', unsafe_allow_html=True)
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="job-col-header">🏢 Job 1</div>', unsafe_allow_html=True)
            t1 = st.text_input("Title 1", placeholder="e.g. Python Developer", key="t1", label_visibility="collapsed")
            d1 = st.text_area("JD 1", height=180, key="d1", placeholder="Paste job description 1…", label_visibility="collapsed")
        with c2:
            st.markdown('<div class="job-col-header">🏢 Job 2</div>', unsafe_allow_html=True)
            t2 = st.text_input("Title 2", placeholder="e.g. Data Analyst", key="t2", label_visibility="collapsed")
            d2 = st.text_area("JD 2", height=180, key="d2", placeholder="Paste job description 2…", label_visibility="collapsed")
        with c3:
            st.markdown('<div class="job-col-header">🏢 Job 3 (Optional)</div>', unsafe_allow_html=True)
            t3 = st.text_input("Title 3", placeholder="e.g. ML Engineer", key="t3", label_visibility="collapsed")
            d3 = st.text_area("JD 3", height=180, key="d3", placeholder="Paste job description 3…", label_visibility="collapsed")
        if st.button("🔀 Compare All Jobs", use_container_width=True, key="btn_matcher"):
            if check_ready():
                if not d1.strip() or not d2.strip():
                    st.warning("Please fill in at least Job 1 and Job 2")
                else:
                    jobs = [{"title": t1 or "Job 1", "desc": d1},
                            {"title": t2 or "Job 2", "desc": d2}]
                    if d3.strip(): jobs.append({"title": t3 or "Job 3", "desc": d3})
                    with st.spinner("Comparing jobs… (~20 seconds)"):
                        result = multi_job_match(st.session_state.resume_text, jobs)
                    st.markdown(result)
                    st.divider()
                    st.download_button("⬇️ Download Comparison", result, "job_comparison.txt", use_container_width=True)

    # ── TAB 7: RAG CHAT ───────────────────────────────────────────────────
    with tabs[6]:
        st.markdown('<div class="sec-header"><div class="sec-title">💬 Chat With Your Resume</div><div class="sec-sub">Ask anything — AI answers based only on your document</div></div>', unsafe_allow_html=True)
        st.divider()

        if not st.session_state.rag_ready:
            st.info("⬅️ Upload your resume in the sidebar (or scan one in the 📸 Camera Scan tab) to activate the chat")
        else:
            st.success("✅ Resume loaded — AI is ready to answer your questions")
            st.markdown("**💡 Quick questions:**")
            chips = ["What are my top skills?", "Summarise my experience",
                     "My strongest achievement?", "Best roles for me?"]
            cols = st.columns(4)
            for col, chip in zip(cols, chips):
                with col:
                    if st.button(chip, key=f"chip_{chip}"):
                        st.session_state._chip = chip
            st.divider()

            for msg in st.session_state.display_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_input = st.chat_input("Ask anything about your resume…")
            if "_chip" in st.session_state:
                user_input = st.session_state.pop("_chip")

            if user_input:
                st.session_state.display_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            answer = st.session_state.rag_chain.invoke({
                                "question": user_input,
                                "chat_history": st.session_state.chat_history,
                            })
                        except Exception as e:
                            answer = f"❌ Error: {e}"
                    st.markdown(answer)
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=answer))
                st.session_state.display_history.append({"role": "assistant", "content": answer})

            if st.session_state.display_history:
                if st.button("🗑️ Clear Chat", key="clear_chat"):
                    st.session_state.chat_history    = []
                    st.session_state.display_history = []
                    st.rerun()

    # ── TAB 8: CAMERA SCAN ────────────────────────────────────────────────
    with tabs[7]:
        render_camera_tab()


# =============================================================================
#  ENTRY POINT  →  streamlit run main.py
# =============================================================================
if __name__ == "__main__":
    main()
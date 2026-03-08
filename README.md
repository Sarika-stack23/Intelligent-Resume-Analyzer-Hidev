# ⚡ Intelligent Resume Analyzer (IRA)

> **AI-powered career platform** — ATS scoring, resume rewriting, cover letters, interview prep, skill roadmaps, multi-job matching, and RAG chat. All in one place.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-6366f1?style=for-the-badge)](https://intelligent-resume-analyzer-hidev-fyqecasubzzh8uytf4n87k.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-1C3C3C?style=for-the-badge)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

---

## 🌐 Live App

**👉 [https://intelligent-resume-analyzer-hidev-fyqecasubzzh8uytf4n87k.streamlit.app/](https://intelligent-resume-analyzer-hidev-fyqecasubzzh8uytf4n87k.streamlit.app/)**

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **ATS Checker** | Match score, keyword analysis, section-by-section scoring |
| ✍️ **Resume Rewriter** | STAR-format bullets, quantified metrics, keyword injection |
| 📝 **Cover Letter Generator** | Three tone styles — Professional, Confident, Creative |
| 🎤 **Interview Prep** | Technical, behavioural & situational questions with STAR answers |
| 🗺️ **Skill Gap Roadmap** | Personalised 90-day learning plan with free resources |
| 🔀 **Multi-Job Matcher** | Compare up to 3 job descriptions, ranked by best fit |
| 💬 **RAG Chat** | Ask anything — AI answers grounded in your own resume |

---

## 🛠️ Tech Stack

- **LLM** — Llama 3.3 70B via [Groq](https://groq.com) (ultra-fast inference)
- **Orchestration** — LangChain 0.3.x (chains, retrievers, prompts)
- **Vector Store** — FAISS (local, in-memory RAG)
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace
- **UI** — Streamlit with custom CSS (Deep Indigo SaaS theme)
- **Document Parsing** — PyPDF + LangChain loaders

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/your-username/intelligent-resume-analyzer.git
cd intelligent-resume-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key
Create a `.env` file in the root:
```env
GROQ_API_KEY=your_groq_api_key_here
```
> Get a free key at [console.groq.com](https://console.groq.com)

### 4. Launch the app
```bash
streamlit run main.py
```

---

## 📁 Project Structure

```
intelligent-resume-analyzer/
├── main.py            # Full app — UI + LLM + RAG logic
├── requirements.txt   # Python dependencies
├── packages.txt       # System packages (for Streamlit Cloud)
├── .env               # API keys (not committed)
└── .gitignore
```

---

## 💡 Usage

1. **Upload your resume** (PDF or TXT) in the sidebar
2. The AI builds a **RAG knowledge base** from your document
3. Navigate the tabs and **paste a job description** in any feature
4. Hit the action button and get instant AI-powered results
5. **Download** any output as a `.txt` file

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key for LLM access |

---

## 📦 Deployment

This app is deployed on **Streamlit Community Cloud**. To deploy your own fork:

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `main.py` as the entry point
4. Add `GROQ_API_KEY` under **Secrets**

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

<div align="center">
  Built with ❤️ using LangChain · Groq · FAISS · Streamlit
</div>

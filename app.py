import os
import zipfile
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Zyro Dynamics | HR AI Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM AESTHETIC CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .hero-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4f46e5 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 28px -6px rgba(79, 70, 229, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
        backdrop-filter: blur(6px);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        opacity: 0.9;
        margin: 0;
        line-height: 1.5;
    }
    .citation-pill {
        display: inline-block;
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        color: #3730a3;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem 0.35rem 0.25rem 0;
        border: 1px solid #c7d2fe;
    }
    .policy-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        color: #4f46e5;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 3px 4px 3px 0;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- API KEYS & TRACING ----------------
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

# ---------------- HERO BANNER ----------------
st.markdown("""
<div class="hero-card">
    <div class="hero-badge">✨ AI-POWERED RAG KNOWLEDGE ASSISTANT</div>
    <div class="hero-title">💼 Zyro Dynamics HR Help Desk</div>
    <div class="hero-subtitle">
        Instant, policy-grounded answers across all 11 official Zyro Dynamics HR documents — complete with page-level source citations and scope guardrails.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ⚡ Quick Sample Questions")
    st.caption("Click any question below to test the bot:")

    sample_questions = [
        "What medical insurance coverage does the company provide?",
        "How much Earned Leave can I carry forward to next year?",
        "Am I eligible to work from home at my grade?",
        "What is the salary range for an L4 Senior employee?",
        "How do I raise a POSH complaint and what's the timeline?",
        "Can you tell me my ESOP vesting schedule?"
    ]

    selected_sample = None
    for q in sample_questions:
        if st.button(f"💬 {q}", use_container_width=True):
            selected_sample = q

    st.markdown("---")
    st.markdown("### 📚 Indexed HR Corpus (11 PDFs)")
    st.markdown("""
    <div>
        <span class="policy-tag">🏢 Company Profile</span>
        <span class="policy-tag">📘 Employee Handbook</span>
        <span class="policy-tag">🌴 Leave Policy</span>
        <span class="policy-tag">🏠 Work From Home</span>
        <span class="policy-tag">⚖️ Code of Conduct</span>
        <span class="policy-tag">📈 Performance Review</span>
        <span class="policy-tag">💰 Compensation & Benefits</span>
        <span class="policy-tag">🔒 IT & Data Security</span>
        <span class="policy-tag">🛡️ POSH Policy</span>
        <span class="policy-tag">🤝 Onboarding & Exit</span>
        <span class="policy-tag">✈️ Travel & Expense</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vector DB", "FAISS")
    with col2:
        st.metric("Retrieval", "MMR (k=6)")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- VECTOR STORE & RETRIEVER LOADER ----------------
@st.cache_resource(show_spinner="Indexing Zyro Dynamics HR Policy PDFs...")
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    if os.path.exists("gen-ai-project-intelligent-rag.zip") and not os.path.exists("extracted_hr_docs"):
        with zipfile.ZipFile("gen-ai-project-intelligent-rag.zip", "r") as z:
            z.extractall("extracted_hr_docs")

    pdf_dir = None
    for root, dirs, files in os.walk("."):
        if any(f.lower().endswith(".pdf") for f in files):
            pdf_dir = root
            break

    if pdf_dir:
        loader = PyPDFDirectoryLoader(pdf_dir)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        return FAISS.from_documents(chunks, embeddings)
    else:
        st.error("⚠️ HR Policy PDFs or ZIP archive not found in repository!")
        st.stop()

# Initialize Vector Store and Retriever
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})

# Initialize Groq LLM with 100% stable current model
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

SYSTEM_PROMPT = """You are an HR policy assistant for Zyro Dynamics.
Rules:
1. Answer ONLY using facts explicitly stated in the context below. Never use outside knowledge or guesses.
2. If a question asks about a policy rule (eligibility, allowance, entitlement, limit, process) but uses 'my' or 'I', state the rule for each grade or band listed in the context. Do NOT refuse just because the grade is unknown.
3. If a question asks for one individual's own records or account data (for example 'my ESOP vesting schedule', 'my leave balance', 'my payslip'), or is outside Zyro Dynamics HR policies (other companies, financials, products, job applications), reply EXACTLY:
"I cannot answer this question as it falls outside the scope of Zyro Dynamics' HR policy documents."
4. Otherwise answer in 1 to 3 clear sentences with the exact numbers, dates, and conditions from the policy.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

# ---------------- CHAT HISTORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Welcome to the Zyro Dynamics HR Help Desk!** Ask me anything about leave entitlements, compensation bands, WFH eligibility, travel reimbursements, or POSH guidelines — or click a sample question in the sidebar to begin.",
            "sources": []
        }
    ]

for msg in st.session_state.messages:
    avatar = "🧑‍💼" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            pills_html = "".join([f'<span class="citation-pill">📄 {src}</span>' for src in msg["sources"]])
            st.markdown(f"<div style='margin-top: 8px;'>{pills_html}</div>", unsafe_allow_html=True)

# ---------------- HANDLE USER INPUT ----------------
user_Input = st.chat_input("Ask about leave, salary bands, WFH, insurance, travel policy...")
active_query = selected_sample if selected_sample else user_Input

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(active_query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching official HR policy documents..."):
            docs = retriever.invoke(active_query)
            context_text = "\n\n".join([
                f"[{os.path.basename(d.metadata.get('source', 'Doc'))} | Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
                for d in docs
            ])
            chain = prompt | llm
            response = chain.invoke({"context": context_text, "question": active_query})
            answer = response.content.strip()

            if "falls outside the scope" in answer:
                st.warning(f"🛡️ **Scope Guardrail Triggered:** {answer}")
            else:
                st.markdown(answer)

            sources = []
            if "falls outside the scope" not in answer:
                for d in docs:
                    src_name = os.path.basename(d.metadata.get("source", "Policy Document")).replace(".pdf", "").replace("_", " ")
                    page_num = d.metadata.get("page", 0) + 1
                    cite = f"{src_name} • p.{page_num}"
                    if cite not in sources:
                        sources.append(cite)
                if sources:
                    pills_html = "".join([f'<span class="citation-pill">📄 {s}</span>' for s in sources])
                    st.markdown(f"<div style='margin-top: 8px;'>{pills_html}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

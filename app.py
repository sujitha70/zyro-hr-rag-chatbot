import os
import re
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------- config
# PDFs sit in the repo root, next to this file
PDF_DIR = Path(__file__).parent
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
REFUSAL = (
    "I cannot answer this question as it falls outside the scope "
    "of Zyro Dynamics' HR policy documents."
)

st.set_page_config(page_title="Zyro Dynamics HR Assistant", page_icon="💬")


# ---------------------------------------------------------------- secrets
def load_secrets():
    """Copy keys from Streamlit secrets into environment variables."""
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("GROQ_API_KEY is missing. Add it in the app's Secrets settings.")
        st.stop()

    # LangSmith tracing is optional
    try:
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"
    except Exception:
        pass


load_secrets()
MODEL_NAME = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b")


# ---------------------------------------------------------------- RAG setup
@st.cache_resource(show_spinner="Loading HR policies and building the index...")
def build_vectorstore():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        st.error("No PDF files found in the repo. Add the HR PDFs next to app.py.")
        st.stop()

    docs = []
    for pdf in pdf_files:
        docs.extend(PyPDFLoader(str(pdf)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings)


@st.cache_resource
def build_chain():
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.7},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an HR policy assistant for Zyro Dynamics.\n"
                "Rules:\n"
                "1. Answer ONLY using facts explicitly stated in the context below. "
                "Never use outside knowledge or guesses.\n"
                "2. Tables in the context may appear flattened into one line. "
                "Read them carefully before deciding the answer is missing.\n"
                "3. If a question asks about a policy rule (eligibility, allowance, "
                "entitlement, limit, process) but uses 'my' or 'I', you do not know "
                "the person's grade, so state the rule for each grade or band listed "
                "in the context. Do NOT refuse just because the grade is unknown.\n"
                "4. If a question asks for one individual's own records or account "
                "data (for example 'my ESOP vesting schedule', 'my leave balance', "
                f"'my payslip'), that data is not in the policy documents, so reply exactly: \"{REFUSAL}\"\n"
                "5. Also reply exactly with that sentence if the context has no "
                "relevant information, or the question is about other companies, "
                "Zyro Dynamics financials, products, job applications, or anything "
                "outside Zyro Dynamics HR policies.\n"
                "6. Otherwise answer in 1 to 3 plain sentences with the exact "
                "numbers, dates and conditions from the policy.\n"
                "7. Plain text only. No markdown, no bullets, no bold.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    llm = ChatGroq(model=MODEL_NAME, temperature=0)
    llm_chain = prompt | llm | StrOutputParser()
    return retriever, llm_chain


def format_docs(docs):
    return "\n\n".join(
        f"[{Path(d.metadata.get('source', '?')).name} | p.{d.metadata.get('page', 0) + 1}]\n"
        f"{d.page_content}"
        for d in docs
    )


def clean(text):
    text = re.sub(r"[*#`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def ask(question):
    retriever, llm_chain = build_chain()
    docs = retriever.invoke(question)
    answer = clean(llm_chain.invoke({"context": format_docs(docs), "question": question}))
    if not answer:
        answer = REFUSAL

    # Citations only for in-scope answers, de-duplicated
    sources = []
    if "outside the scope" not in answer:
        seen = set()
        for d in docs:
            name = Path(d.metadata.get("source", "?")).name
            page = d.metadata.get("page", 0) + 1
            if (name, page) not in seen:
                seen.add((name, page))
                sources.append({"document": name, "page": page})
    return answer, sources


def show_sources(sources):
    if sources:
        with st.expander("Source Citations"):
            for s in sources:
                st.write(f"- {s['document']} (page {s['page']})")


# ---------------------------------------------------------------- UI
st.title("Zyro Dynamics HR Policy Assistant")
st.caption("Ask questions about leave, WFH, compensation, conduct, travel and other HR policies.")

with st.sidebar:
    st.header("About")
    st.write(
        "Answers come only from the Zyro Dynamics HR policy documents. "
        "Questions outside those documents are politely declined."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Build the index once at startup (cached)
build_chain()

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources", []))

# New question
if question := st.chat_input("Ask an HR policy question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            try:
                answer, sources = ask(question)
            except Exception as e:
                answer, sources = f"Sorry, something went wrong: {str(e)[:200]}", []
        st.write(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

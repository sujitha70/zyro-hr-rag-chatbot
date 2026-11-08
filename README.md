# 💼 Zyro Dynamics HR Help Desk — Intelligent RAG Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-00599C)
![Groq](https://img.shields.io/badge/LLM-Llama_3.3_70B_(Groq)-F55036)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangSmith](https://img.shields.io/badge/Observability-LangSmith-orange)

An AI-powered **Retrieval-Augmented Generation (RAG)** HR Help Desk built for **Zyro Dynamics**. This assistant answers employee queries across **11 official HR policy documents** with strict factual grounding, page-level source citations, and automated out-of-scope refusal guardrails.

---

## ✨ Key Features

* **📚 Multi-Document Policy Indexing:** Automatically extracts and indexes all 11 official Zyro Dynamics HR policy PDFs using `PyPDFDirectoryLoader` and `RecursiveCharacterTextSplitter` (`chunk_size=800`, `chunk_overlap=150`).
* **🔍 Dense Semantic Search with MMR:** Uses HuggingFace `BAAI/bge-small-en-v1.5` embeddings stored in a **FAISS** vector database with **Maximal Marginal Relevance (MMR)** (`k=6`, `fetch_k=20`) to retrieve relevant, non-redundant policy sections.
* **⚡ Ultra-Fast LLM Inference:** Powered by **Groq (`llama-3.3-70b-versatile`)** with `temperature=0` for deterministic, policy-accurate responses.
* **🛡️ Strict Scope & Privacy Guardrails:**
  * Answers policy rules across employee grades/bands when users ask general eligibility questions using *"my"* or *"I"*.
  * Automatically refuses requests for personal employee records (e.g., personal ESOP schedules, individual leave balances, payslips) or non-HR topics with a standardized guardrail response.
* **📄 Page-Level Source Citations:** Every in-scope response displays interactive pill badges citing the exact policy PDF name and page number.
* **📊 End-to-End Observability:** Integrated with **LangSmith (`zyro-rag-challenge`)** to trace latency, token usage, retrieved document chunks, and guardrail adherence.

---

## 🗂️ Indexed HR Policy Corpus (11 Documents)

1. **Company Profile & Overview**
2. **Employee Handbook**
3. **Leave & Attendance Policy**
4. **Work From Home (WFH) Policy**
5. **Code of Conduct & Ethics**
6. **Performance Management & Appraisal Policy**
7. **Compensation, Grades & Benefits Policy**
8. **IT Acceptable Use & Data Security Policy**
9. **Prevention of Sexual Harassment (POSH) Policy**
10. **Onboarding, Probation & Exit Policy**
11. **Domestic & International Travel & Expense Policy**

---

## 🏗️ RAG Architecture

```text
Employee Query
      │
      ▼
[BAAI/bge-small-en-v1.5 Embeddings]
      │
      ▼
[FAISS Vector Store] ──► MMR Retriever (Top k=6 diverse chunks from 20 candidates)
      │
      ▼
[Prompt Template + Scope Guardrail Rules + Source Metadata]
      │
      ▼
[Groq LLM: llama-3.3-70b-versatile (temp=0)]
      │
      ├──► In-Scope Query     ──► Grounded 1-3 Sentence Answer + PDF & Page Citations
      └──► Out-of-Scope Query ──► Standardized Guardrail Refusal Alert

# tutor_rag_app.py
"""
Pathfinder Tutor — Browser-based RAG Tutor (1st grade → graduation)
Uses Streamlit for a web UI.
"""

import os
import uuid
from dataclasses import dataclass
from typing import Dict, Any, List

import streamlit as st
from dotenv import load_dotenv
from textwrap import dedent
import nest_asyncio
nest_asyncio.apply()

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser


# ================= PROMPTS =================
BASE_SYSTEM_PROMPT = dedent("""
You are "Pathfinder", a friendly AI tutor. Adapt explanations to the learner's grade level,
subject, and learning style. Always:
- Be accurate and cite passages from provided context.
- If no relevant context is provided or the context is empty, answer the question using your general knowledge.
- Adjust depth, vocabulary, and examples to the learner’s grade.
- Encourage understanding with short checks-for-understanding.

Sections to include:
1) Explanation
2) Key idea(s)
3) Short example
4) Quick check
5) Sources
""").strip()

GRADE_GUIDANCE = {
    "1": "Use very simple words and short sentences. Prefer concrete examples.",
    "2": "Use simple words, short sentences, and lots of examples.",
    "3": "Keep it simple but introduce basic terms.",
    "4": "Use simple terms and real-life examples.",
    "5": "Increase detail slightly; use bullet points.",
    "6": "Explain clearly with definitions and analogies.",
    "7": "Use middle-school vocabulary. Encourage reasoning.",
    "8": "Use clear structure; blend examples with definitions.",
    "9": "High-school freshman level; multiple perspectives briefly.",
    "10": "High-school level; concise reasoning steps.",
    "11": "Pre-college level; structured argument and formulas if needed.",
    "12": "Advanced high-school; precise terminology with intuition.",
    "undergrad": "Undergraduate level; formal definitions and references.",
    "postgrad": "Graduate level; formalism, assumptions, citations."
}

LEARNING_STYLE_GUIDANCE = {
    "visual": "Use diagrams-in-words, spatial descriptions, bullet lists.",
    "verbal": "Use clear, sequential paragraphs and concise definitions.",
    "kinesthetic": "Suggest hands-on mini-activities or experiments.",
    "logical": "Emphasize structure, rules, patterns, proofs, or algorithms.",
    "social": "Frame with collaborative examples.",
    "solitary": "Offer self-paced steps and reflection prompts."
}


def build_instruction(grade_level: str, subject: str, learning_style: str) -> str:
    grade = grade_level.lower().strip()
    style = learning_style.lower().strip()
    grade_tip = GRADE_GUIDANCE.get(grade, GRADE_GUIDANCE.get("10"))
    style_tip = LEARNING_STYLE_GUIDANCE.get(style, "Use clear structure and examples.")
    subject_line = f"Subject focus: {subject}." if subject else "Subject focus: general."
    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"Grade level: {grade_level}\n"
        f"{subject_line}\n"
        f"Style guidance: {style_tip}\n"
        f"Level guidance: {grade_tip}\n"
    )


# ================= INGEST =================
DATA_DIR = "data"
INDEX_DIR = "index_faiss"

def load_all_docs():
    docs = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            path = os.path.join(root, f)
            ext = f.lower()
            try:
                if ext.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    # Limit to first 20 pages to avoid memory issues on Vercel
                    docs += loader.load()[:20] if len(loader.load()) > 20 else loader.load()
                elif ext.endswith(".txt") or ext.endswith(".md"):
                    docs += TextLoader(path, encoding="utf-8").load()
                elif ext.endswith(".docx"):
                    docs += UnstructuredWordDocumentLoader(path).load()
                elif ext.endswith(".html") or ext.endswith(".htm"):
                    docs += UnstructuredHTMLLoader(path).load()
                else:
                    st.warning(f"Skipping unsupported file: {path}")
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    return docs


def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    docs = load_all_docs()
    if not docs:
        print("No documents found in ./data. Add PDFs/TXT/MD/DOCX/HTML and rerun.")
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=10, separators=["\n\n", "\n", " ", ""]
    )
    splits = splitter.split_documents(docs)

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.from_documents(splits, embedding=embeddings)
    vectordb.save_local(INDEX_DIR)
    return True


# ================= TUTOR =================
@dataclass
class TutorConfig:
    grade_level: str = "10"
    subject: str = "General"
    learning_style: str = "verbal"
    top_k: int = 5
    temperature: float = 0.2
    model: str = os.getenv("RAG_CHAT_MODEL", "gemini-1.5-flash")


def format_docs(docs: List[Document]) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", None)
        loc = f" (p.{page})" if page is not None else ""
        out.append(f"[{i}] Source: {os.path.basename(src)}{loc}\n{d.page_content.strip()}\n")
    return "\n\n".join(out)


def build_chain(cfg: TutorConfig, session_id: str):
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": cfg.top_k})

    system_text = build_instruction(cfg.grade_level, cfg.subject, cfg.learning_style)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Question: {question}\n\nContext:\n{context}\n")
    ])

    llm = ChatGoogleGenerativeAI(model=cfg.model, temperature=cfg.temperature)

    def _retrieve(inputs: Dict[str, Any]) -> Dict[str, Any]:
        q = inputs["question"]
        docs = retriever.get_relevant_documents(q)
        return {"context": format_docs(docs), "question": q}

    rag = (
        RunnablePassthrough.assign(context=RunnableLambda(lambda x: _retrieve(x)["context"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    # Session-based memory
    if "histories" not in st.session_state:
        st.session_state.histories = {}
    if session_id not in st.session_state.histories:
        st.session_state.histories[session_id] = ChatMessageHistory()

    with_history = RunnableWithMessageHistory(
        rag,
        lambda _: st.session_state.histories[session_id],
        input_messages_key="question",
        history_messages_key="history"
    )
    return with_history


# ================= STREAMLIT UI =================
def main():
    st.set_page_config(page_title="Pathfinder Tutor", page_icon="📚", layout="wide")
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not set in .env. Please add GOOGLE_API_KEY=your_api_key to your .env file.")
        return

    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    st.title("📚 Pathfinder Tutor (RAG)")
    st.write("Ask questions from 1st grade to graduation. Upload study material into the `data/` folder.")

    if not os.path.isdir(INDEX_DIR):
        with st.spinner("Building knowledge index..."):
            if not build_index():
                return

    with st.sidebar:
        st.header("⚙️ Settings")
        grade = st.text_input("Grade level", "10")
        subject = st.text_input("Subject", "General")
        style = st.selectbox(
            "Learning style",
            ["verbal", "visual", "kinesthetic", "logical", "social", "solitary"],
            index=0
        )
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.1)

    cfg = TutorConfig(
        grade_level=grade, subject=subject, learning_style=style, temperature=temperature
    )
    session_id = "default-session"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask a question...")
    if user_input:
        chain = build_chain(cfg, session_id)
        with st.spinner("Thinking..."):
            answer = chain.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}}
            )
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", answer))

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(msg)
        else:
            st.chat_message("assistant").markdown(msg)


if __name__ == "__main__":
    main()

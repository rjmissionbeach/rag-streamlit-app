import io
import re
import time
from typing import Dict, List, Tuple

import requests
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Document RAG Chat", page_icon="📄", layout="wide")

DEFAULT_MODEL_NAME = "openrouter/free"
INSUFFICIENT_INFO_MESSAGE = "I don't have enough information in the uploaded documents to answer that."

NUMBER_PATTERN = re.compile(r"\$\s?[\d,]{3,}|\b\d{1,3}(?:,\d{3})+\b")
FIGURE_WORDS = re.compile(
    r"\b(how much|how many|total|revenue|profit|income|margin|ebitda|"
    r"units sold|earnings|cost|expense|price|value|amount|percent|%)\b",
    re.IGNORECASE,
)


# -----------------------------
# Document processing
# -----------------------------
def extract_text(filename: str, filebytes: bytes) -> str:
    """Extract text from PDF, HTML, or plain-text files."""
    fname_lower = filename.lower()

    if fname_lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(filebytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if fname_lower.endswith((".html", ".htm")):
        soup = BeautifulSoup(filebytes, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Preserve table rows so labels and values stay attached.
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                cells = [c for c in cells if c]
                if cells:
                    row.replace_with(" | ".join(cells) + "\n")

        return soup.get_text(separator="\n")

    return filebytes.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 250) -> List[str]:
    """Split text into overlapping character chunks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step

    return [c for c in chunks if len(c.strip()) > 50]


def financial_density(text: str) -> float:
    """Reward chunks with many financial figures."""
    matches = NUMBER_PATTERN.findall(text)
    return min(len(matches) / 10, 1.0)


@st.cache_data(show_spinner=False)
def build_index_from_files(
    uploaded_files_data: Tuple[Tuple[str, bytes], ...],
    chunk_size: int,
    overlap: int,
) -> Dict:
    """Extract text, chunk documents, and build a TF-IDF index."""
    all_chunks: List[str] = []
    chunk_sources: List[str] = []
    file_summaries = []

    for filename, filebytes in uploaded_files_data:
        text = extract_text(filename, filebytes)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        chunk_sources.extend([filename] * len(chunks))
        file_summaries.append({"filename": filename, "chunks": len(chunks)})

    if not all_chunks:
        return {
            "ready": False,
            "error": "No usable text was extracted from the uploaded files.",
            "file_summaries": file_summaries,
        }

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    chunk_vectors = vectorizer.fit_transform(all_chunks)
    chunk_financial_scores = [financial_density(c) for c in all_chunks]

    return {
        "ready": True,
        "all_chunks": all_chunks,
        "chunk_sources": chunk_sources,
        "vectorizer": vectorizer,
        "chunk_vectors": chunk_vectors,
        "chunk_financial_scores": chunk_financial_scores,
        "file_summaries": file_summaries,
    }


# -----------------------------
# Retrieval and generation
# -----------------------------
def retrieve(
    question: str,
    index: Dict,
    top_k: int = 10,
    min_score: float = 0.03,
    financial_boost: float = 0.15,
) -> List[Dict]:
    """Retrieve relevant chunks using TF-IDF plus optional financial-figure boost."""
    q_vec = index["vectorizer"].transform([question])
    tfidf_scores = cosine_similarity(q_vec, index["chunk_vectors"])[0]
    wants_figure = bool(FIGURE_WORDS.search(question))

    combined_scores = []
    for i, base_score in enumerate(tfidf_scores):
        score = float(base_score)
        if wants_figure:
            score += financial_boost * index["chunk_financial_scores"][i]
        combined_scores.append(score)

    ranked = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
    results = []

    for i in ranked[:top_k]:
        if combined_scores[i] >= min_score:
            results.append(
                {
                    "text": index["all_chunks"][i],
                    "source": index["chunk_sources"][i],
                    "score": combined_scores[i],
                }
            )

    return results


def ask_openrouter(
    question: str,
    passages: List[Dict],
    api_key: str,
    model_name: str,
    max_retries: int = 3,
) -> str:
    """Send retrieved context and the user question to OpenRouter."""
    if not passages:
        return INSUFFICIENT_INFO_MESSAGE

    context = "\n\n".join(f"[Source: {p['source']}]\n{p['text']}" for p in passages)

    system_prompt = (
        "You answer questions using ONLY the provided context passages. "
        "Some passages contain financial tables where values may appear in sequence near "
        "their row/column labels (e.g. 'Total revenues | $20,322 | $13,673' means the first "
        "number follows the first year column, etc.) — match values to labels carefully. "
        f"If the passages don't contain enough information to answer the question, say exactly: "
        f"'{INSUFFICIENT_INFO_MESSAGE}' "
        "Do not use outside knowledge. Do not guess or speculate."
    )

    user_prompt = f"""
You are a financial research assistant.

Answer the user's question using ONLY the supplied context.

Instructions:
- Give a direct answer first.
- Use specific numbers when available.
- Be concise.
- If the answer is not contained in the context, say so.
- Do not copy large chunks of text from the source documents.
- Synthesize information from multiple passages when appropriate.

Context passages:

{context}

Question:
{question}
"""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "Document RAG Chat",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=45,
            )
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                continue
            return "Request timed out after multiple attempts. Try again or switch models."
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                continue
            return f"Network error: {e}"

        if response.status_code == 200:
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            sources_used = sorted(set(p["source"] for p in passages))
            return f"{answer}\n\nSources consulted: {', '.join(sources_used)}"

        if response.status_code == 429:
            wait = 30
            try:
                wait = int(response.json()["error"]["metadata"]["retry_after_seconds"])
            except (KeyError, ValueError, TypeError):
                pass
            if attempt < max_retries - 1:
                time.sleep(wait)
                continue

        return f"API error ({response.status_code}): {response.text}"

    return "Failed after retries — either rate-limited or a network issue. Try again shortly, or switch models."


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Document RAG Chat")
st.caption("Upload PDF, TXT, or HTML files, then ask questions using only those documents.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenRouter API key", type="password")
    model_name = st.text_input("Model", value=DEFAULT_MODEL_NAME)

    st.divider()
    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "html", "htm"],
        accept_multiple_files=True,
    )

    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Passages to retrieve", min_value=3, max_value=20, value=10)
    min_score = st.slider("Minimum retrieval score", min_value=0.00, max_value=0.20, value=0.05, step=0.01)
    financial_boost = st.slider("Financial-number boost", min_value=0.00, max_value=0.50, value=0.15, step=0.05)

    with st.expander("Advanced chunking"):
        chunk_size = st.number_input("Chunk size", min_value=500, max_value=3000, value=1200, step=100)
        overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=250, step=50)

    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if not uploaded_files:
    st.info("Upload one or more documents in the sidebar to begin.")
    st.stop()

uploaded_files_data = tuple((f.name, f.getvalue()) for f in uploaded_files)

with st.spinner("Building document index..."):
    index = build_index_from_files(uploaded_files_data, int(chunk_size), int(overlap))

if not index.get("ready"):
    st.error(index.get("error", "Could not build the document index."))
    st.stop()

with st.sidebar:
    st.success(f"Indexed {len(index['all_chunks'])} chunks from {len(uploaded_files)} file(s).")
    with st.expander("Indexed files"):
        for item in index["file_summaries"]:
            st.write(f"- {item['filename']}: {item['chunks']} chunks")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("passages"):
            with st.expander("Retrieved passages"):
                for p in message["passages"]:
                    st.markdown(f"**{p['source']}** — score `{p['score']:.3f}`")
                    st.text(p["text"][:1000])

question = st.chat_input("Ask a question about the uploaded documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            answer = "Enter your OpenRouter API key in the sidebar before asking questions."
            passages = []
            st.markdown(answer)
        else:
            passages = retrieve(
                question,
                index=index,
                top_k=top_k,
                min_score=min_score,
                financial_boost=financial_boost,
            )
            with st.spinner("Retrieving passages and asking the model..."):
                answer = ask_openrouter(question, passages, api_key, model_name)
            st.markdown(answer)

            with st.expander("Retrieved passages"):
                if passages:
                    for p in passages:
                        st.markdown(f"**{p['source']}** — score `{p['score']:.3f}`")
                        st.text(p["text"][:1000])
                else:
                    st.write("No passages met the retrieval threshold.")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "passages": passages}
    )

import streamlit as st
import ollama
import numpy as np
from pypdf import PdfReader


# -----------------------------------
# 🔹 Chunking
# -----------------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# -----------------------------------
# 🔹 Create Embeddings
# -----------------------------------
def create_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk
        )
        embeddings.append(response["embedding"])
    return embeddings


# -----------------------------------
# 🔹 Cosine Similarity
# -----------------------------------
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# -----------------------------------
# 🔹 Retrieval
# -----------------------------------
def retrieve_chunks(question, chunks, embeddings, k=3):

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=question
    )
    question_embedding = response["embedding"]

    similarities = []
    for idx, chunk_embedding in enumerate(embeddings):
        score = cosine_similarity(question_embedding, chunk_embedding)
        similarities.append((score, idx))

    similarities.sort(reverse=True)

    top_chunks = []
    for i in range(min(k, len(similarities))):
        top_chunks.append(chunks[similarities[i][1]])

    return top_chunks


# -----------------------------------
# 🔹 Build RAG Prompt
# -----------------------------------
def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question:
{question}
"""

    return prompt


# -----------------------------------
# 🔹 Streamlit Setup
# -----------------------------------
st.set_page_config(page_title="Local PDF RAG Chatbot", page_icon="📄")
st.title("📄 Local PDF RAG Chatbot (Ollama RAG)")

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")


# -----------------------------------
# 🔹 Session State
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = []


# -----------------------------------
# 🔹 PDF Processing
# -----------------------------------
if uploaded_file is not None:

    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    chunks = chunk_text(text)
    st.session_state.chunks = chunks

    st.sidebar.success("PDF loaded & chunked!")
    st.sidebar.write(f"Total Chunks: {len(chunks)}")

    with st.spinner("Creating embeddings..."):
        embeddings = create_embeddings(chunks)
        st.session_state.embeddings = embeddings

    st.sidebar.success("Embeddings created!")
    st.sidebar.write(f"Embedding Dimension: {len(embeddings[0])}")


# -----------------------------------
# 🔹 Display Chat
# -----------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Ask something about your PDF...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.chunks and st.session_state.embeddings:

        # 🔥 Retrieval
        retrieved_chunks = retrieve_chunks(
            prompt,
            st.session_state.chunks,
            st.session_state.embeddings,
            k=3
        )

        # 🔥 Build RAG prompt
        rag_prompt = build_prompt(prompt, retrieved_chunks)

        # 🔥 Call LLM with context
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": rag_prompt}]
        )

        answer = response["message"]["content"]

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

    else:
        with st.chat_message("assistant"):
            st.write("Please upload and process a PDF first.")



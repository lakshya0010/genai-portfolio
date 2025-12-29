import streamlit as st
import re
import ollama
from sentence_transformers import SentenceTransformer
import chromadb


def chunk_text(text,max_size = 300):
    sentences = re.split(r'(?<=[!.?])\s+', text.strip())
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if(len(sentence)+len(current_chunk) < max_size):
            current_chunk += sentence + " "

        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


st.title("RAG Document Q&A")
st.write("Upload a document and get relevent questions answered")

uploaded_file = st.file_uploader("Upload a .txt document", type=["txt"])

if uploaded_file:
    document_text = uploaded_file.read().decode("utf-8")
    chunks = chunk_text(document_text)

    st.write(f'document split into {len(chunks)} Chunks')

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).tolist()

    clint = chromadb.Client()
    collection = clint.get_or_create_collection(name="rag_1")

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": f"chunk_{i}"} for i in range(len(chunks))]
    )

    question = st.text_input("Ask a question: ")
    
    if st.button("Ask") and question:
        query_embedding = model.encode([question]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=2
        )

        retrieved_docs = results["documents"][0]
        retrieved_sources = [m["source"] for m in results["metadatas"][0]]

        context = '\n'.join(retrieved_docs)

        prompt = f"""
Use the following context to answer the question.
If the answer is not in the context, say "Not found in the document."

Context:
{context}

Question:
{question}
"""
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role":"system", "content":"You answer only from context and do not invent facts."},
                {"role":"user", "content": prompt}
            ]
        )

        st.subheader("Answer")
        st.write(response["message"]["content"])

        st.subheader("Sources")
        for src in retrieved_sources:
            st.write("-", src)
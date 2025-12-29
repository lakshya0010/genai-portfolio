import re
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import psutil

document_text = """
Machine learning is a field of artificial intelligence that focuses on building systems that learn from data.
It is widely used in areas such as recommendation systems, fraud detection, and image recognition.
Deep learning is a subset of machine learning that uses neural networks with many layers.
Natural language processing allows machines to understand and generate human language.
Large language models are trained on massive amounts of text data.
These models work with tokens rather than raw text.
Because of token limits, long documents cannot be passed entirely to an LLM.
Instead, documents are split into smaller chunks.
Each chunk should be small enough to fit within context limits.
Chunking also helps retrieval systems find relevant parts of documents.
Good chunking improves the quality of answers in RAG systems.
Poor chunking can lead to missing context or incomplete answers.
Therefore, choosing the right chunk size is an important design decision.
"""

def sentence_chunk_text(text, max_size=300):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunk = []
    current_chunk = ""
    for sentence in sentences:
        if(len(current_chunk)+len(sentence)<=max_size):
            current_chunk += sentence + " "
        else:
            chunk.append(current_chunk)
            current_chunk = sentence + " "
    if(current_chunk):
        chunk.append(current_chunk)
    return chunk
chunks = sentence_chunk_text(document_text)

# for idx, chunk in enumerate(chunks):
#     print(f'Chunk {idx+1}  length: {len(chunk)}')
#     print(chunk.strip())
    
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()

chroma_clint = chromadb.Client()
collection = chroma_clint.get_or_create_collection(name="rag_demo")

collection.add(
    documents=chunks,
    embeddings= embeddings,
    ids=[f'Chunk_{i}' for i in range(len(chunks))],
    metadatas=[{"source": f"Chunk_{i}"} for i in range(len(chunks))]
)

query = "Why is chunking important in RAG Systems?"

query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

retrieved_documents = results["documents"][0]
retrieved_sources = [m["source"] for m in results["metadatas"][0]]

context = "\n".join(retrieved_documents)


prompt = f"""
Use the following context to answer the question.
If the answer is not in the context, say "Not found in the document."

Context:
{context}

Question:
{query}

Answer the question using only the context.
Do NOT mention sources or citations in your answer.

"""



def has_enough_ram(required_gb: float) -> bool:
    available = psutil.virtual_memory().available / (1024**3)
    return available > required_gb

if not has_enough_ram(4.0):
    raise RuntimeError("Not enough RAM to load this model safely")



response = ollama.chat(
    model="llama3",
    messages=[
        {"role": "system", "content": "You are a precise assistant that cites sources."},
        {"role": "user", "content": prompt}
    ]
)

print("\nAnswer: ")
print(response["message"]["content"])
print("\nSources:", retrieved_sources)







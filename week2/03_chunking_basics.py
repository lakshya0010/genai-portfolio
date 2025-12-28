import re

long_text = """
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


def naive_chunk_text(long_text):
    chunks = []
    limit = 0
    for idx, ch in enumerate(long_text):
        if idx - limit == 300:
            chunks.append(long_text[limit:idx])
            limit = idx

        if (idx == len(long_text)-1):
            chunks.append(long_text[limit:idx])

    return chunks

naive_chunks = naive_chunk_text(long_text)
print("NAIVE CHUNKING")
print("Total chunks:", len(naive_chunks))
for i, chunk in enumerate(naive_chunks):
    print(f"\nChunk {i+1} (length {len(chunk)}):")
    print(chunk.strip())


def sentence_chunk_text(text, max_size = 300):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunk = []
    current_chunk = ""
    for sentence in sentences:
        if(len(sentence)+len(current_chunk) <= max_size):
            current_chunk += sentence + " "
        else:
            chunk.append(current_chunk)
            current_chunk = sentence
    if(current_chunk):
        chunk.append(current_chunk)
    return chunk

sentence_chunks = sentence_chunk_text(long_text)
print("SENTENCE CHUNKING")
print("Total chunks:", len(sentence_chunks))
for i, chunk in enumerate(sentence_chunks):
    print(f"\nChunk {i+1} (length {len(chunk)}):")
    print(chunk.strip())



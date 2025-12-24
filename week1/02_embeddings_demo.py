from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love machine learning",
    "Machine learning is amazing",
    "I enjoy eating pizza",
    "The weather is nice today"
]

# Generate embeddings
embeddings = model.encode(sentences)

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)

# Print first embedding (first 10 values)
print("First embedding vector (first 10 values):")
print(embeddings[0][:10])

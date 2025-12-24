# week1/03_semantic_similarity.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love machine learning",
    "Machine learning is amazing",
    "I enjoy eating pizza",
    "The weather is nice today"
]

# Generate embeddings
embeddings = model.encode(sentences)

# Compare similarities
sim_01 = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)[0][0]

sim_02 = cosine_similarity(
    [embeddings[0]],
    [embeddings[2]]
)[0][0]

sim_03 = cosine_similarity(
    [embeddings[0]],
    [embeddings[3]]
)[0][0]

print("Similarity: ML vs ML:", sim_01)
print("Similarity: ML vs Pizza:", sim_02)
print("Similarity: ML vs Weather:", sim_03)

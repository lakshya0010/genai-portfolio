from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer("all-MiniLM-L6-v2")

corpus = [
    "Machine learning is a field of artificial intelligence.",
    "Deep learning is a subset of machine learning.",
    "Natural language processing deals with text data.",
    "Pizza is a popular Italian food.",
    "The weather today is sunny and pleasant.",
    "Transformers are powerful models for NLP tasks.",
    "Embeddings represent text as numerical vectors."
]

corpus_embeddings = model.encode(corpus)

def semantic_search(query, top_k = 3):
    query_embedding = model.encode(query)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = cos_scores.topk(k=top_k)

    print(f"\nQuery: {query}")
    print("Top results: ")
    
    for score, idx in zip(top_results[0], top_results[1]):
        print(f"Score: {score:.4f} | Text: {corpus[idx]}")


semantic_search("What are embeddings?")
semantic_search("Tell me about machine learning")
semantic_search("I like Italian food")
import chromadb
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="demo_collection")


corpus = [
    "Machine learning is a field of artificial intelligence.",
    "Deep learning is a subset of machine learning.",
    "Natural language processing deals with text data.",
    "Pizza is a popular Italian food.",
    "The weather today is sunny and pleasant.",
    "Transformers are powerful models for NLP tasks.",
    "Embeddings represent text as numerical vectors."
]

for idx, sentence in enumerate(corpus):
    collection.add(
        documents = [sentence],
        ids = [f"id{idx}"]
    )

results = collection.query(
    query_texts=["What is Machine Learning"],
    n_results=3
)

print("Results: ")
for doc in results["documents"][0]:
    print(doc)
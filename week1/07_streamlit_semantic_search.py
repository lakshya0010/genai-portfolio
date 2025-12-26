import streamlit as st
import chromadb 
from sentence_transformers import SentenceTransformer

st.title("Semantic Search")
st.write("Type a query to retrieve semantically similar sentences.")

client = chromadb.Client()
collection = client.get_or_create_collection(name="demo-collection")

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
    
query = st.text_input("Enter your query: ")

if query:
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    st.subheader("Results")
    for doc in results["documents"][0]:
        st.write("-", doc)

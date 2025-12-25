# week1 - embeddings & nlp
## Day 1 Notes

- embeddings: numerical vector representations of text that capture semantic meaning
- semantic similarity: measures similarity in meaning using cosine similarity between embeddings
- tokenization: splitting text into tokens that models can process
- preprocessing: cleaning text (lowercasing, removing urls, splitting sentences)
- tf-idf vs embeddings: tf-idf is keyword-based, embeddings capture meaning

## Day 2 Notes

- Loaded the SentenceTransformer model: all-MiniLM-L6-v2
all-MiniLM-L6-v2 is a compact, general-purpose sentence embedding model optimized for fast, reasonably accurate semantic similarity across common NLP tasks.
- Converted sentences into fixed-length vectors (384 dimensional)
- The output vectors represent the semantic meaning of sentences


## Day 3 Notes

- Used cosine similarity to compare sentence embeddings
- Semantically similar sentences have higher cosine similarity scores
- This principle is used in semantic search and RAG retrieval


## Day 4 Notes

- Text is broken into tokens before being processed by models
- Token count is different from word count due to subword tokenization
- Token limits affect prompt size, cost, and chunking strategies


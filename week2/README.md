# week2 - vector dbs & semantic search

## Day 8 Notes

- Used a local LLM (Llama 3 via Ollama)
- Sent system and user prompts programmatically
- Observed how prompt instructions control output style


## Day 9 Notes

- Learned common prompt engineering patterns
- Observed how role, constraints, and examples affect output
- Understood why prompt structure is critical for reliable LLM behavior


## Day 10 Notes

- Learned why long documents must be chunked for LLMs
- Implemented naive and sentence-aware chunking
- Understood how chunk size affects retrieval and context



## Day 11 Notes

- Built a complete RAG pipeline using local LLMs
- Combined chunking, embeddings, vector DB retrieval, and generation
- Observed grounded answers based on document context


## Day 12 Notes

- Tuned retrieval quality using different top-k values
- Added metadata to chunks for citation support
- Implemented simple source attribution to reduce hallucinations


## Day 13 Notes

- Built a Streamlit-based RAG application
- Allowed users to upload documents and ask questions
- Displayed grounded answers with source attribution
- A little less stable, need to work on that
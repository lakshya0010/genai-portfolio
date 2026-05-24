What are embeddings?
Embeddings are numerical vector representations of data—most commonly text—that capture semantic meaning in a form machines can efficiently process and compare.

Why do we need vector databases?
We need vector databases because once data is converted into embeddings, traditional databases become fundamentally inefficient and incorrect for working with them at scale.

What is chunking, and why does it matter?
Chunking is the process of splitting large pieces of data—most commonly text—into smaller, semantically coherent units before creating embeddings and storing them in a vector database.
It matters because LLMs do not retrieve documents — they retrieve chunks. The quality of retrieval, grounding, and final answers depends directly on how well chunking is done.

What problem does RAG solve?
RAG fixes this by grounding generation in retrieved, external knowledge at query time.

How does RAG reduce hallucination?
RAG reduces hallucination by changing what the LLM is allowed to rely on when generating an answer.
Hallucination occurs when a model is forced to guess. RAG works by removing the need to guess.

Difference between RAG and fine-tuning
RAG adds knowledge at query time. Fine-tuning changes model behavior at training time.
They operate at different layers of the system.

What is an agent?
An agent is an LLM wrapped with memory, tools, and control logic so it can act, not just respond.

Why are uncontrolled agents risky?
Uncontrolled agents are risky because they can take autonomous actions without sufficient constraints, verification, or oversight, leading to incorrect, unsafe, or costly outcomes.

A controlled agent is an autonomous system that can act, but only within clearly defined limits, with checks at every decision and execution point.

Where does LangChain fit in?
LangChain fits in as an agent and application orchestration framework—it is infrastructure glue, not an AI model and not a database.




I am a B.Tech Mathematics & Computing student at DTU with hands-on experience building GenAI and backend systems using Python.

I recently built a controlled GenAI automation agent exposed via a FastAPI service and would be excited to apply similar skills while learning from your team.

I have attached my resume and GitHub link for reference.

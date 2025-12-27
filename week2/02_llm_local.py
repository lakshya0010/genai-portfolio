import ollama

print("Local LLM ready!")

response = ollama.chat(
    model = "llama3",
    messages=[
        {"role":"system", "content": "You are a helpful assistant. Answer in one short sentence."},
        {"role":"user", "content": "What are embeddings?"}
    ]

)

print(response["message"]["content"])
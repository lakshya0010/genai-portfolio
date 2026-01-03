import ollama

#Tools
def search_web(query):
    return(
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "information retrieval with language models to produce factual, grounded answers. "
        "It retrieves relevant documents and uses them as context for generation."
    )

def extract_key_points(text):
    sentences = text.split(".")
    return[
        sentences[0].strip(),
        sentences[1].strip(),
        sentences[2].strip()
    ]

def summarize(text):
    response = ollama.chat(
        model="llama3",
        messages=[{"role":"system", "content": "Summarize clearly in 2–3 sentences. give only the summary and no additional text"},
                  {"role":"user", "content": text}
                  ]
    )
    return response["message"]["content"]


#Agent
def research_agent(topic):
    state = {
        "search_results": None,
        "key_points": None,
        "summary": None
    }

    state["search_results"] = search_web(topic)
    state["key_points"] = extract_key_points(state["search_results"])

    combined_points = " ".join(state["key_points"])
    state["summary"] = summarize(combined_points)
    return state



topic = "Rag systems"
result = research_agent(topic)
print("\n=== Research Report ===\n")
print("Topic:", topic)
print("\nKey Points:")
for p in result["key_points"]:
    print("-", p)

print("\nSummary:")
print(result["summary"])



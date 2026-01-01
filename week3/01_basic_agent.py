import ollama


#Tools
def search_docs(query):
    return f"Search results for {query}: AI, ML and RAG concepts."

def summarize(text):
    return text[:60] + "..."


def choose_tool(user_input):
    prompt = f"""
You are an agent.

You MUST respond in EXACTLY this format:
tool:<tool_name>
input:<tool_input>

Do NOT add explanations.
Do NOT add extra text.
Do NOT change the format.

Available tools:
- search_docs(query)
- summarize(text)

User input:
{user_input}
"""

    
    response = ollama.chat(
        model="llama3",
        messages=[{"role":"user", "content": prompt}]
    )

    return response["message"]["content"]


def parse_decision(decision_text):
    tool = None
    tool_input = None

    for line in decision_text.splitlines():
        if line.lower().startswith("tool:"):
            tool = line.split(":", 1)[1].strip()
        elif line.lower().startswith("input:"):
            tool_input = line.split(":", 1)[1].strip()

    if tool is None or tool_input is None:
        raise ValueError(f"Could not parse decision:\n{decision_text}")

    return tool, tool_input


user_input = "Summarize: RAG systems reduce hallucinations by using retrieved context."
decision = choose_tool(user_input)
print("Decision:\n", decision)

tool_name, tool_input = parse_decision(decision)

if tool_name == "search_docs":
    result = search_docs(tool_input)
elif tool_name == "summarize":
    result = summarize(tool_input)
else:
    result = "No valid tool selected."

final_prompt = f"""
User question:
{user_input}

Tool output:
{result}

Provide a helpful final response.
"""

final_response = ollama.chat(
    model="llama3",
    messages=[{"role":"user", "content": final_prompt}]
)

print(final_response["message"]["content"])
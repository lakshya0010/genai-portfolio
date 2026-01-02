import ollama

#Tools
def search_web(query):
    return f"Search results about '{query}': RAG improves facutal accuracy using retrieval"

def summarize(text):
    return text[:80] + "..."

def extract_key_points(text):
    sentences = text.split(".")
    points = [s.strip() for s in sentences if s.strip()][:3]
    return points


#Agent decision
def choose_action(user_input, memory):
    prompt = f"""
You are an intelligent agent.

Available tools:
- search_web(query)
- summarize(text)
- extract_key_points(text)
- finish(answer)

Current memory:
{memory}

User input:
{user_input}

Decide the next action.

Respond ONLY in this format:
tool:<tool_name>
input:<tool_input>

You should use tool:finish when:
- the user request has been fully satisfied
- OR repeating a tool would not add new information

If you choose tool:finish,
the input MUST be the final answer to the user.
Do NOT explain your reasoning.
Do NOT say None.
Do NOT mention memory.

"""
    
    response = ollama.chat(
        model="llama3",
        messages=[{"role":"user", "content":prompt}]
    )

    return response["message"]["content"]

#Parsing
def parse_action(text):
    tool = None
    tool_input = None

    for line in text.splitlines():
        if line.lower().startswith("tool:"):
            tool = line.split(":",1)[1].strip()
        elif line.lower().startswith("input:"):
            tool_input = line.split(":",1)[1].strip()

    if not tool or tool_input is None:
        raise ValueError(f"Invalid agent output:\n{text}")
    
    return tool, tool_input


#Agent loop
user_input = "Find information about RAG systems and summarize it"
agent_memory = {}
max_steps = 3
used_tools = set()

for step in  range(max_steps):
    print(f"\n---Step {step+1}---")

    decision = choose_action(user_input,agent_memory)
    print("agent decision:\n", decision)

    try:
        tool,tool_input = parse_action(decision)
    except Exception as e:
        print("Parsing failed", e)
        break

    if tool in used_tools:
        tool = "finish"
        tool_input = agent_memory[f"step_{step}"]["output"]
    used_tools.add(tool)

    if tool == "finish" or len(used_tools) >= 2:
        final_answer = agent_memory[f"step_{step}"]["output"]
        print("\nFinal Answer:")
        print(final_answer)
        break

    if tool == "search_web":
        output = search_web(tool_input)
    elif tool == "summarize":
        output = summarize(tool_input)
    elif tool == "extract_key_points":
        output = extract_key_points(tool_input)
    else:
        print("Unknown tool selected.")
        break


    agent_memory[f"step_{step+1}"] = {
        "tool" : tool,
        "input" : tool_input,
        "output" : output
    }

    print("Tool output:", output)

else:
    print("\nAgent stopped due to step limit.")
    
        

from langchain_community.llms import Ollama
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = Ollama(model="llama3")


@tool
def classify_input(text :str) -> str:
    """Classify input as email, task or other."""
    prompt = f"""
Classify the following input into ONE category:
email, task, or other.

Respond with only ONE word, the ONE category.
Input:
{text}
"""
    return llm.invoke(prompt)

@tool
def extract_intent(text:str)->str:
    """Extract the intent from the input."""
    prompt = f"""
Extract the intent from the following input.
Respond with bullet points.

Input:
{text}
"""
    return llm.invoke(prompt)

@tool 
def generate_email_reply(text:str)->str:
    """Generate a professional email reply."""
    prompt = f"""
Write a clear, professional email reply to the following message.

Message:
{text}
"""
    return llm.invoke(prompt)

@tool
def generate_task_summary(text:str)->str:
    """Generate a task summary."""
    prompt = f"""
Convert the following into task summary using the following bullet points.

Input:
{text}
"""
    return llm.invoke(prompt)


def run_agent(user_input:str):
    result = {}

    input_type = classify_input.invoke(user_input).strip().lower()
    result["type"] = input_type

    intent = extract_intent.invoke(user_input).strip().lower()
    result["intent"] = intent

    if input_type == "email":
        output = generate_email_reply.invoke(intent)
    elif input_type == "task":
        output = generate_task_summary.invoke(intent)
    else:
        output = "No automation action required."
    
    result["output"] = output
    return result

if __name__ == "__main__":
    user_input = """
    Hi, can you please send the project update by tomorrow evening?
    Also let me know if you need any help.
    """

    result = run_agent(user_input)

    print("\nTYPE:")
    print(result["type"])
    print("\nIntent:")
    print(result["intent"])
    print("\nOutput:")
    print(result["output"])



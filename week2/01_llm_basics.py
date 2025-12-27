from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("API key not found. Check .env file")

print("API Key successfully loaded.")

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer in one short sentence."
        },
        {
            "role": "user",
            "content": "What are embeddings?"
        }
    ]
)
print(response.output_text)
from fastapi import FastAPI
from pydantic import BaseModel

from week3.langchain_email_agent import run_agent

app = FastAPI()


class InputText(BaseModel):
    text:str


@app.post("/process")
def process_text(input_data: InputText):
    result = run_agent(input_data.text)
    return result

# week1/01_text_cleaning.py
import re
from typing import List

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)              # collapse whitespace
    text = re.sub(r'http\S+', '', text)           # remove urls
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)   # keep basic punctuation
    text = text.strip()
    return text

def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

if __name__ == "__main__":
    sample = """Hello! This is a sample sentence. Visit https://example.com for more info.
    NLP & embeddings are awesome.  """
    cleaned = clean_text(sample)
    print("CLEANED:", cleaned)
    print("SENTENCES:", split_sentences(cleaned))

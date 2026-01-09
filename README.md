# GenAI Agent System (Email & Task Automation)

This repository contains a production-style GenAI system that classifies text inputs (emails or tasks), extracts intent, and generates structured outputs.  
The system is built using a controlled LangChain agent, exposed via a FastAPI backend and consumed through a Streamlit frontend.

---

## 🚀 Main Project

### Email & Task Automation Agent

**Capabilities**
- Classifies raw text as email, task, or other
- Extracts intent before generation
- Generates structured responses
- Controlled agent flow (no autonomous loops)
- API-first design with frontend client


## Architecture

User
↓
Streamlit UI
↓
FastAPI (/process)
↓
LangChain Agent
↓
Local LLM (Ollama)




**Tech Stack**
- Python
- LangChain
- Ollama (local LLM)
- FastAPI
- Streamlit

---

## 🧠 Project Structure

week0/ → environment setup & git

week1/ → embeddings & NLP fundamentals

week2/ → vector databases & RAG

week3/ → agent logic & tools

week4/ → API & frontend (production system)


The earlier weeks build foundational components used in the final system.

---

## ▶️ How to Run the Main Project

### 1. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Start the API
```
uvicorn week4.agent_api:app --reload
```

### 4. Start the UI
```
streamlit run week4/streamlit_client.py
```

Limitations

Uses a local LLM (performance depends on hardware)

No authentication or persistent memory

Designed for learning and demonstration purposes



👤 Author

Lakshya Sharma
LinkedIn: https://www.linkedin.com/in/lakshya-sharma-551583312





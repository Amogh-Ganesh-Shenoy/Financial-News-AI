# Phase 2 — Financial Report RAG Chatbot 🤖

An AI-powered chatbot that reads financial PDF reports and answers 
questions using Retrieval Augmented Generation (RAG).

## What it does
- Loads any financial PDF report
- Splits it into searchable chunks
- Converts chunks into vector embeddings
- Finds the most relevant sections for your question
- Answers using only the document — no hallucination

## Built With
- Python 3.11
- OpenAI GPT-3.5-turbo
- LangChain
- FAISS Vector Store
- PyPDF

## How RAG works

Your question
↓
Converted to numbers (embeddings)
↓
FAISS finds 3 most relevant chunks
↓
Chunks + question sent to GPT
↓
GPT answers using only the document

## How to run it

**1. Install dependencies**

pip install langchain langchain-openai langchain-community faiss-cpu pypdf langchain-text-splitters langchain-core

**2. Add your API key to .env**

OPENAI_API_KEY=your_key_here

**3. Add your PDF report**

Place any financial PDF in the phase2 folder and name it `report.pdf`

**4. Run the chatbot**

python rag_chatbot.py

## Example Questions
- "What was the revenue for Q4 2025?"
- "What were the key highlights of the report?"
- "What is the earnings per share?"

## Example Output
You: What was the revenue for Q4 2025?
AI: The revenue for Q4 2025 was $3.916 billion.
You: What is the earnings per share?
AI: For full-year 2025:
Basic: $14.67
Diluted: $14.66

## Part of the Financial News AI Project
- Phase 1 — Live financial news summariser
- Phase 2 — Financial PDF RAG chatbot (this)
- Phase 3 — Full deployed financial intelligence app (coming soon)

## Author
Amogh Ganesh Shenoy
[GitHub](https://github.com/Amogh-Ganesh-Shenoy)

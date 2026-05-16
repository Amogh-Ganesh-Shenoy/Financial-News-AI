# Phase 3 — JPMorgan Financial Intelligence App 🏦

A full-stack AI-powered financial intelligence dashboard focused on 
JPMorgan Chase, combining live market data, RAG-powered chat, and 
advanced financial analytics.

## Features

### 📈 Live Market Data
- Real-time JPMorgan stock price and key metrics
- 12-month price and volume history chart
- Annual revenue and net income charts
- Live data sourced from Yahoo Finance

### 🤖 Filing Intelligence
- RAG-powered chat over live JPMorgan financial data
- Ask questions in plain English, get precise answers
- Grounded in real financial figures — no hallucination
- Fixed bottom chat input like a professional chat app

### 📊 Financial Analytics
- Price with 50-day and 200-day moving averages
- Quarterly revenue and net income breakdown
- Key financial ratios — ROE, ROA, profit margin
- Analyst recommendations visualised

## Built With
- Python 3.11
- OpenAI GPT-3.5-turbo
- LangChain + FAISS (RAG)
- yfinance (live financial data)
- Streamlit (web application)
- Plotly (interactive charts)

## How to run locally

**1. Clone the repository**

git clone https://github.com/Amogh-Ganesh-Shenoy/Financial-News-AI.git

**2. Install dependencies**

pip install streamlit openai langchain langchain-openai langchain-community langchain-text-splitters langchain-core faiss-cpu yfinance plotly pandas python-dotenv

**3. Add your API key**

Create a `.env` file in the root folder:

OPENAI_API_KEY=your_openai_key_here

**4. Run the app**

cd phase3
streamlit run app.py

## Part of the Financial News AI Project
- Phase 1 — Live financial news AI summariser
- Phase 2 — Financial PDF RAG chatbot
- Phase 3 — Full JPMorgan financial intelligence app (this)

## Author
Amogh Ganesh Shenoy
[GitHub](https://github.com/Amogh-Ganesh-Shenoy)


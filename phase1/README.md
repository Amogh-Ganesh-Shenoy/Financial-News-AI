# Financial News AI Summariser 📈

An AI-powered financial news analyser built with Python and OpenAI GPT.
Fetches real-time financial news and returns professional analysis instantly.

## What it does
- Fetches live financial news from around the world
- Summarises each article in 2 clear sentences
- Detects market sentiment — Bullish, Bearish, or Neutral
- Identifies key companies and stocks mentioned
- Explains why each story matters to investors

## Built With
- Python 3.11
- OpenAI GPT-3.5-turbo
- NewsAPI
- Pandas
- Requests

## How to run it

**1. Clone the repository**
git clone https://github.com/Amogh-Ganesh-Shenoy/Financial-News-AI.git

**2. Install dependencies**
pip install openai python-dotenv requests pandas streamlit

**3. Add your API keys**

Create a `.env` file in the root folder:

OPENAI_API_KEY=your_openai_key_here
NEWS_API_KEY=your_newsapi_key_here

**4. Run the analyser**

python summariser.py

## Example Output

ARTICLE 1: Bitcoin, Nasdaq investors are celebrating...
Summary: Bitcoin and Nasdaq are surging but consumer
sentiment is at record lows, showcasing a disconnect
between financial markets and the real economy.
Sentiment: Neutral
Key companies: Bitcoin, Nasdaq
Why it matters: Investors should monitor the growing gap
between market performance and consumer sentiment.

## Author
Amogh Ganesh Shenoy  
[GitHub](https://github.com/Amogh-Ganesh-Shenoy)
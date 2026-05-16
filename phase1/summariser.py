import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarise_article(title, description):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """You are a financial news analyst. 
                When given a news headline and description return:
                1. A 2 sentence summary
                2. Sentiment: Bullish, Bearish, or Neutral
                3. Key companies or stocks mentioned
                4. Why this matters to investors (1 sentence)"""
            },
            {
                "role": "user",
                "content": f"Headline: {title}\n\nDescription: {description}"
            }
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def get_financial_news():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "stock market finance earnings",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": os.getenv("NEWS_API_KEY")
    }
    response = requests.get(url, params=params)
    return response.json()

# Fetch and analyse real news
print("Fetching latest financial news...\n")
print("=" * 60)

news = get_financial_news()

for i, article in enumerate(news["articles"], 1):
    title = article["title"]
    description = article["description"] or "No description available"
    
    print(f"\nARTICLE {i}: {title}")
    print("-" * 60)
    analysis = summarise_article(title, description)
    print(analysis)
    print("=" * 60)
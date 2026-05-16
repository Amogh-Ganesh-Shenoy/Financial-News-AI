import os
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import tempfile
import requests
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sec_edgar_downloader import Downloader
import glob
import time

# ─────────────────────────────────────
# SETUP
# ─────────────────────────────────────
load_dotenv(find_dotenv())
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="JPMorgan Financial Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar and footer
st.markdown("""
    <style>
    [data-testid="collapsedControl"] {display: none}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Hide streamlit footer */
    .st-emotion-cache-1cypcdb {display: none;}
    /* Push content up so it clears the fixed input */
    .main .block-container {
    padding-bottom: 120px;
}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        font-size: 16px;
        font-weight: 500;
    }
    /* Fix chat input to bottom */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem 2rem;
        background: #0e1117;
        z-index: 999;
        border-top: 1px solid #2d2d2d;
    }
    /* Add padding at bottom so messages don't hide behind input */
    .stChatMessage {
        margin-bottom: 0.5rem;
    }
    [data-testid="stChatMessageContainer"] {
        padding-bottom: 100px;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# HEADER
# ─────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 🏦")
with col_title:
    st.title("JPMorgan Financial Intelligence")
    st.markdown("""
<div style='text-align: left;'>
<small>Powered by OpenAI GPT-3.5-turbo · LangChain · FAISS · Streamlit</small><br>
<small>Data sources: Yahoo Finance (yfinance) · SEC EDGAR</small>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────
# TABS
# ─────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈 Live Market Data",
    "🤖 Filing Intelligence",
    "📊 Financial Analytics"
])

# ─────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────
@st.cache_data(ttl=300)
def get_jpm_data():
    """Fetch live JPMorgan data from yfinance. Cached for 5 minutes."""
    ticker = yf.Ticker("JPM")
    info = ticker.info
    history = ticker.history(period="1y")
    financials = ticker.financials
    quarterly = ticker.quarterly_financials
    recommendations = ticker.recommendations
    return info, history, financials, quarterly, recommendations

def get_jpm_financial_text():
    """Convert live yfinance JPMorgan data into clean text for RAG."""
    ticker = yf.Ticker("JPM")
    info = ticker.info
    financials = ticker.financials
    quarterly = ticker.quarterly_financials
    balance_sheet = ticker.balance_sheet

    text = f"""
JPMORGAN CHASE & CO. FINANCIAL INTELLIGENCE REPORT
Generated from Yahoo Finance Live Data

COMPANY OVERVIEW
================
Company: {info.get('longName', 'JPMorgan Chase')}
Sector: {info.get('sector', 'Financial Services')}
Industry: {info.get('industry', 'Banking')}
Employees: {info.get('fullTimeEmployees', 0):,}

CURRENT MARKET DATA
===================
Current Price: ${info.get('currentPrice', 0):.2f}
Previous Close: ${info.get('previousClose', 0):.2f}
Market Cap: ${info.get('marketCap', 0)/1e9:.2f} Billion
52 Week High: ${info.get('fiftyTwoWeekHigh', 0):.2f}
52 Week Low: ${info.get('fiftyTwoWeekLow', 0):.2f}

VALUATION METRICS
=================
P/E Ratio Trailing: {info.get('trailingPE', 0):.2f}
P/E Ratio Forward: {info.get('forwardPE', 0):.2f}
Price to Book: {info.get('priceToBook', 0):.2f}
Earnings Per Share: ${info.get('trailingEps', 0):.2f}

PROFITABILITY
=============
Profit Margin: {info.get('profitMargins', 0)*100:.2f}%
Return on Equity: {info.get('returnOnEquity', 0)*100:.2f}%
Return on Assets: {info.get('returnOnAssets', 0)*100:.2f}%
Operating Margins: {info.get('operatingMargins', 0)*100:.2f}%

DIVIDENDS
=========
Dividend Rate: ${info.get('dividendRate', 0):.2f}
Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%
Payout Ratio: {info.get('payoutRatio', 0)*100:.2f}%

ANALYST RECOMMENDATIONS
=======================
Recommendation: {info.get('recommendationKey', 'N/A').upper()}
Target Mean Price: ${info.get('targetMeanPrice', 0):.2f}
Target High Price: ${info.get('targetHighPrice', 0):.2f}
Target Low Price: ${info.get('targetLowPrice', 0):.2f}
Number of Analysts: {info.get('numberOfAnalystOpinions', 0)}
"""

    # Add annual financials
    if not financials.empty:
        text += "\nANNUAL FINANCIAL STATEMENTS\n"
        text += "===========================\n"
        for col in financials.columns:
            text += f"\nYear: {str(col)[:10]}\n"
            for idx in financials.index:
                val = financials.loc[idx, col]
                if pd.notna(val):
                    text += f"  {idx}: ${val/1e9:.2f}B\n"

    # Add quarterly financials
    # Add quarterly financials — more searchable format
    if not quarterly.empty:
       text += "\nQUARTERLY FINANCIAL STATEMENTS\n"
       text += "================================\n"
       quarters = list(quarterly.columns)
       for col in quarterly.columns:
           quarter_label = str(col)[:10]
           text += f"\nQuarterly Results for period ending {quarter_label}:\n"
           for idx in quarterly.index:
               val = quarterly.loc[idx, col]
               if pd.notna(val):
                  text += (
                       f"  JPMorgan quarterly {idx} "
                       f"for {quarter_label}: ${val/1e9:.2f} Billion\n"
                        )

    # Add explicit quarter comparisons
       if len(quarters) >= 2:
          text += "\nQUARTER OVER QUARTER COMPARISON\n"
          text += "=================================\n"
          for idx in quarterly.index:
              text += f"\n{idx} by quarter:\n"
              for col in quarterly.columns:
                  val = quarterly.loc[idx, col]
                  if pd.notna(val):
                    text += f"  {str(col)[:10]}: ${val/1e9:.2f}B\n"

    # Add balance sheet
    if not balance_sheet.empty:
        text += "\nBALANCE SHEET\n"
        text += "=============\n"
        for col in balance_sheet.columns:
            text += f"\nDate: {str(col)[:10]}\n"
            for idx in balance_sheet.index:
                val = balance_sheet.loc[idx, col]
                if pd.notna(val):
                    text += f"  {idx}: ${val/1e9:.2f}B\n"

    return text

def get_sec_filing():
    """Fetch latest JPMorgan 10-Q directly from SEC EDGAR API."""
    headers = {
        "User-Agent": "FinancialAI research@financialai.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov"
    }

    try:
        # JPMorgan CIK is 0000019617
        # Get list of recent filings
        url = "https://data.sec.gov/submissions/CIK0000019617.json"
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            st.error(f"SEC API returned status: {response.status_code}")
            return None

        data = response.json()
        filings = data["filings"]["recent"]

        forms = filings["form"]
        accession_numbers = filings["accessionNumber"]
        primary_docs = filings["primaryDocument"]

        # Find latest 10-Q
        latest_accession = None
        latest_primary_doc = None

        for i, form in enumerate(forms):
            if form == "10-Q":
                latest_accession = accession_numbers[i].replace("-", "")
                latest_primary_doc = primary_docs[i]
                break

        if not latest_accession:
            st.error("No 10-Q filing found")
            return None

        # Fetch the actual document
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"19617/{latest_accession}/{latest_primary_doc}"
        )

        st.info(f"Fetching: {doc_url}")

        doc_headers = {
            "User-Agent": "FinancialAI research@financialai.com"
        }
        doc_response = requests.get(
            doc_url, headers=doc_headers, timeout=30
        )

        if doc_response.status_code != 200:
            st.error(f"Document fetch failed: {doc_response.status_code}")
            return None

        # Clean HTML
        import re
        text = doc_response.text
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        clean_text = re.sub(r'&[a-zA-Z]+;', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text[:50000]

    except Exception as e:
        st.error(f"SEC filing error: {e}")
        return None

def build_rag_from_text(text):
    """Build RAG chain from text content."""
    

    # Create document
    doc = Document(page_content=text, metadata={"source": "JPM 10-Q SEC Filing"})

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents([doc])

    # Build vector store
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Build prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a senior financial analyst specialising in JPMorgan Chase.
    Answer using only the SEC filing context below.
    Be specific — cite numbers, percentages, and figures where available.
    If not found say: "This information is not in the current filing."

    Context: {context}
    Question: {question}
    Answer:
    """)

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def format_large_number(num):
    """Format large numbers into readable format."""
    if num is None:
        return "N/A"
    if abs(num) >= 1_000_000_000:
        return f"${num/1_000_000_000:.2f}B"
    if abs(num) >= 1_000_000:
        return f"${num/1_000_000:.2f}M"
    return f"${num:,.2f}"

# ─────────────────────────────────────
# TAB 1 — LIVE MARKET DATA
# ─────────────────────────────────────
with tab1:
    st.subheader("📈 JPMorgan Chase — Live Market Data")
    st.caption("Data sourced from Yahoo Finance · Updates every 5 minutes")

    with st.spinner("Fetching live JPMorgan data..."):
        try:
            info, history, financials, quarterly, recommendations = get_jpm_data()

            # Key metrics row
            st.subheader("Key Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)

            current_price = info.get("currentPrice", 0)
            prev_close = info.get("previousClose", 0)
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100

            with m1:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                )
            with m2:
                st.metric(
                    "Market Cap",
                    format_large_number(info.get("marketCap"))
                )
            with m3:
                st.metric(
                    "52W High",
                    f"${info.get('fiftyTwoWeekHigh', 0):.2f}"
                )
            with m4:
                st.metric(
                    "52W Low",
                    f"${info.get('fiftyTwoWeekLow', 0):.2f}"
                )
            with m5:
                st.metric(
                    "P/E Ratio",
                    f"{info.get('trailingPE', 0):.2f}"
                )

            st.markdown("---")

            # Stock price chart
            st.subheader("12-Month Stock Price History")
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=history.index,
                y=history["Close"],
                mode="lines",
                name="JPM Close Price",
                line=dict(color="#1f77b4", width=2)
            ))
            fig_price.add_trace(go.Bar(
                x=history.index,
                y=history["Volume"],
                name="Volume",
                yaxis="y2",
                opacity=0.3,
                marker_color="#aec7e8"
            ))
            fig_price.update_layout(
                title="JPMorgan Chase (JPM) — Price & Volume",
                yaxis=dict(title="Price (USD)"),
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right"
                ),
                hovermode="x unified",
                height=450
            )
            st.plotly_chart(fig_price, use_container_width=True)

            st.markdown("---")

            # Financial highlights
            st.subheader("Annual Financial Highlights")
            if not financials.empty:
                fin_data = financials.T
                fin_data.index = pd.to_datetime(fin_data.index).year

                col_f1, col_f2 = st.columns(2)

                with col_f1:
                    if "Total Revenue" in fin_data.columns:
                        fig_rev = px.bar(
                            x=fin_data.index,
                            y=fin_data["Total Revenue"] / 1e9,
                            title="Annual Revenue (Billions USD)",
                            labels={"x": "Year", "y": "Revenue ($B)"},
                            color_discrete_sequence=["#1f77b4"]
                        )
                        st.plotly_chart(fig_rev, use_container_width=True)

                with col_f2:
                    if "Net Income" in fin_data.columns:
                        fig_income = px.bar(
                            x=fin_data.index,
                            y=fin_data["Net Income"] / 1e9,
                            title="Annual Net Income (Billions USD)",
                            labels={"x": "Year", "y": "Net Income ($B)"},
                            color_discrete_sequence=["#2ca02c"]
                        )
                        st.plotly_chart(fig_income, use_container_width=True)

            # Company info
            st.markdown("---")
            st.subheader("About JPMorgan Chase")
            st.info(info.get("longBusinessSummary", "No description available."))

        except Exception as e:
            st.error(f"Error fetching data: {e}")

# ─────────────────────────────────────
# TAB 2 — FILING INTELLIGENCE
# ─────────────────────────────────────
with tab2:
    st.subheader("🤖 JPMorgan SEC Filing Intelligence")
    st.caption("RAG-powered chat over live JPMorgan financial data from Yahoo Finance")

    if st.button("📥 Load JPMorgan Financial Intelligence", type="primary"):
        with st.spinner("Building JPMorgan knowledge base from live data..."):
            filing_text = get_jpm_financial_text()

            if filing_text is None:
                st.error("Could not retrieve SEC filing. Try again.")
            else:
                with st.spinner("Building RAG knowledge base..."):
                    st.session_state["rag_chain"] = build_rag_from_text(
                        filing_text
                    )
                    st.session_state["filing_loaded"] = True
                    st.session_state["chat_history"] = []
                st.success("✅ JPMorgan financial intelligence loaded and ready!")
                st.info(f"📄 Loaded {len(filing_text):,} characters of live financial data")

    if st.session_state.get("filing_loaded"):
        st.markdown("---")
        st.subheader("💬 Ask questions about JPMorgan's latest filing")
        st.caption("Questions are answered using only the official SEC document")

        # Suggested questions
        st.markdown("**Suggested questions:**")
        sq1, sq2, sq3 = st.columns(3)
        with sq1:
             if st.button("What was net income?"):
                 st.session_state["suggested_q"] = "What was JPMorgan's net income?"
        with sq2:
             if st.button("What is return on equity?"):
                 st.session_state["suggested_q"] = "What is JPMorgan's return on equity and what does it mean?"
        with sq3:
              if st.button("What was revenue growth?"):
                  st.session_state["suggested_q"] = "What was JPMorgan's revenue growth across quarters?"
        # Display chat history
        for message in st.session_state.get("chat_history", []):
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Handle suggested question
        auto_q = st.session_state.pop("suggested_q", None)

        # Chat input
        question = st.chat_input("Ask anything about JPMorgan's SEC filing...")

        if question or auto_q:
            q = auto_q or question
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                with st.spinner("Searching filing..."):
                    answer = st.session_state["rag_chain"].invoke(q)
                    st.write(answer)
            st.session_state["chat_history"].append(
                {"role": "user", "content": q}
            )
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer}
            )
    else:
        st.info("👆 Click the button above to load JPMorgan's latest SEC filing.")

# ─────────────────────────────────────
# TAB 3 — FINANCIAL ANALYTICS
# ─────────────────────────────────────
with tab3:
    st.subheader("📊 JPMorgan Financial Analytics")
    st.caption("Deep financial analysis powered by yfinance data")

    with st.spinner("Loading analytics..."):
        try:
            info, history, financials, quarterly, recommendations = get_jpm_data()

            # Moving averages
            st.subheader("Price & Moving Averages")
            history["MA50"] = history["Close"].rolling(window=50).mean()
            history["MA200"] = history["Close"].rolling(window=200).mean()

            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(
                x=history.index, y=history["Close"],
                name="Price", line=dict(color="#1f77b4", width=1.5)
            ))
            fig_ma.add_trace(go.Scatter(
                x=history.index, y=history["MA50"],
                name="50-day MA", line=dict(color="#ff7f0e", width=1.5)
            ))
            fig_ma.add_trace(go.Scatter(
                x=history.index, y=history["MA200"],
                name="200-day MA", line=dict(color="#2ca02c", width=1.5)
            ))
            fig_ma.update_layout(
                title="JPM Price with Moving Averages",
                yaxis_title="Price (USD)",
                hovermode="x unified",
                height=400
            )
            st.plotly_chart(fig_ma, use_container_width=True)

            st.markdown("---")

            # Quarterly financials
            st.subheader("Quarterly Financial Performance")
            if not quarterly.empty:
                q_data = quarterly.T
                q_data.index = q_data.index.astype(str)

                col_q1, col_q2 = st.columns(2)

                with col_q1:
                    if "Total Revenue" in q_data.columns:
                        fig_qrev = px.bar(
                            x=q_data.index,
                            y=q_data["Total Revenue"] / 1e9,
                            title="Quarterly Revenue ($B)",
                            labels={"x": "Quarter", "y": "Revenue ($B)"},
                            color_discrete_sequence=["#1f77b4"]
                        )
                        st.plotly_chart(fig_qrev, use_container_width=True)

                with col_q2:
                    if "Net Income" in q_data.columns:
                        fig_qinc = px.bar(
                            x=q_data.index,
                            y=q_data["Net Income"] / 1e9,
                            title="Quarterly Net Income ($B)",
                            labels={"x": "Quarter", "y": "Net Income ($B)"},
                            color_discrete_sequence=["#2ca02c"]
                        )
                        st.plotly_chart(fig_qinc, use_container_width=True)

            st.markdown("---")

            # Key financial ratios
            st.subheader("Key Financial Ratios")
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Return on Equity",
                         f"{info.get('returnOnEquity', 0)*100:.2f}%")
            with r2:
                st.metric("Return on Assets",
                         f"{info.get('returnOnAssets', 0)*100:.2f}%")
            with r3:
                st.metric("Profit Margin",
                         f"{info.get('profitMargins', 0)*100:.2f}%")
            with r4:
                st.metric("Dividend Yield",
                         f"{info.get('dividendYield', 0)*100:.2f}%")

            st.markdown("---")

            # Analyst recommendations
            st.subheader("Analyst Recommendations")
            if recommendations is not None and not recommendations.empty:
                rec_counts = recommendations.iloc[-20:][
                    "strongBuy strongSell buy hold sell".split()
                ].sum()
                fig_rec = px.bar(
                    x=rec_counts.index,
                    y=rec_counts.values,
                    title="Analyst Recommendations (Last 20)",
                    labels={"x": "Recommendation", "y": "Count"},
                    color=rec_counts.index,
                    color_discrete_map={
                        "strongBuy": "#00C853",
                        "buy": "#69F0AE",
                        "hold": "#FFD600",
                        "sell": "#FF6D00",
                        "strongSell": "#D50000"
                    }
                )
                st.plotly_chart(fig_rec, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading analytics: {e}")


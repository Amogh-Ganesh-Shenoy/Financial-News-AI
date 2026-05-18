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
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Live Market Data",
    "🤖 Filing Intelligence",
    "📊 Financial Analytics",
    "🧮 Investment Calculator"
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

# ─────────────────────────────────────
# TAB 4 — INVESTMENT CALCULATOR + CHAT
# ─────────────────────────────────────
with tab4:
    st.subheader("🧮 JPMorgan Investment Calculator")
    st.caption(
        "Real calculations using live JPMorgan data — "
        "ask about YOUR money, not just the company"
    )

    with st.spinner("Loading live data..."):
        try:
            info, history, financials, quarterly, recommendations = (
                get_jpm_data()
            )

            current_price = info.get("currentPrice", 0)
            dividend_yield = info.get("dividendYield", 0) or 0
            dividend_rate = info.get("dividendRate", 0) or 0
            target_mean = info.get("targetMeanPrice", 0) or 0
            target_high = info.get("targetHighPrice", 0) or 0
            target_low = info.get("targetLowPrice", 0) or 0
            pe_ratio = info.get("trailingPE", 0) or 0
            eps = info.get("trailingEps", 0) or 0

            # Calculate 12 month actual return
            if not history.empty:
                price_12m_ago = history["Close"].iloc[0]
                price_now = history["Close"].iloc[-1]
                actual_12m_return = (
                    (price_now - price_12m_ago) / price_12m_ago
                ) * 100
            else:
                actual_12m_return = 0

            total_annual_return = (
                actual_12m_return + (dividend_yield * 100)
            )

            # Live data summary
            st.subheader("📊 Live Data Powering the Calculators")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("Current Price", f"${current_price:.2f}")
            with d2:
                st.metric("Dividend Yield",
                         f"{dividend_yield*100:.2f}%")
            with d3:
                st.metric("12M Price Return",
                         f"{actual_12m_return:.2f}%")
            with d4:
                st.metric("Total Annual Return",
                         f"{total_annual_return:.2f}%")

            st.markdown("---")

            # CALCULATOR 1
            st.subheader("💰 Calculator 1 — Investment Return Estimator")
            st.caption(
                "Based on JPMorgan's actual 12-month price performance"
            )

            col_c1, col_c2 = st.columns([1, 1])
            with col_c1:
                investment_amount = st.number_input(
                    "How much would you like to invest? (USD)",
                    min_value=100,
                    max_value=10000000,
                    value=10000,
                    step=100,
                    key="inv_amount"
                )
                time_horizon = st.selectbox(
                    "Time horizon",
                    ["3 months", "6 months",
                     "1 year", "2 years", "5 years"],
                    index=2
                )

            with col_c2:
                horizon_map = {
                    "3 months": 0.25,
                    "6 months": 0.5,
                    "1 year": 1,
                    "2 years": 2,
                    "5 years": 5
                }
                years = horizon_map[time_horizon]
                annual_return = actual_12m_return / 100
                projected_value = investment_amount * (
                    (1 + annual_return) ** years
                )
                profit_loss = projected_value - investment_amount
                total_return_pct = (
                    (projected_value - investment_amount) /
                    investment_amount
                ) * 100
                annual_dividend = investment_amount * dividend_yield
                total_dividend = annual_dividend * years
                total_with_dividend = projected_value + total_dividend

                st.metric(
                    "Projected Value (price only)",
                    f"${projected_value:,.2f}",
                    f"{profit_loss:+,.2f} ({total_return_pct:+.2f}%)"
                )
                st.metric(
                    "Dividend Income over period",
                    f"${total_dividend:,.2f}"
                )
                st.metric(
                    "Total Value (price + dividends)",
                    f"${total_with_dividend:,.2f}",
                    f"+${total_with_dividend - investment_amount:,.2f}"
                )

            st.markdown("---")

            # CALCULATOR 2
            st.subheader("💵 Calculator 2 — Dividend Income Calculator")
            st.caption(
                "How much passive income would your investment generate?"
            )

            col_d1, col_d2 = st.columns([1, 1])
            with col_d1:
                div_investment = st.number_input(
                    "Investment amount (USD)",
                    min_value=100,
                    max_value=10000000,
                    value=10000,
                    step=100,
                    key="div_investment"
                )

            with col_d2:
                annual_div_income = div_investment * dividend_yield
                monthly_div_income = annual_div_income / 12
                weekly_div_income = annual_div_income / 52
                shares = (
                    div_investment / current_price
                    if current_price > 0 else 0
                )
                st.metric("Shares you would own",
                         f"{shares:.2f} shares")
                st.metric("Annual dividend income",
                         f"${annual_div_income:,.2f}")
                st.metric("Monthly dividend income",
                         f"${monthly_div_income:,.2f}")
                st.metric("Weekly dividend income",
                         f"${weekly_div_income:,.2f}")

            if dividend_yield > 0:
                years_to_double = 72 / (dividend_yield * 100)
                st.info(
                    f"📌 **Rule of 72:** At {dividend_yield*100:.2f}% "
                    f"dividend yield, it would take approximately "
                    f"**{years_to_double:.1f} years** to double your "
                    f"investment through dividends alone."
                )

            st.markdown("---")

            # CALCULATOR 3
            st.subheader(
                "🎯 Calculator 3 — Analyst Price Target Calculator"
            )
            st.caption(
                "Based on Wall Street analyst consensus targets"
            )

            col_t1, col_t2 = st.columns([1, 1])
            with col_t1:
                target_investment = st.number_input(
                    "Investment amount (USD)",
                    min_value=100,
                    max_value=10000000,
                    value=10000,
                    step=100,
                    key="target_investment"
                )
                st.markdown(f"""
                **Analyst Targets:**
                - 🎯 Mean: **${target_mean:.2f}**
                - 📈 High: **${target_high:.2f}**
                - 📉 Low: **${target_low:.2f}**
                - 💵 Current: **${current_price:.2f}**
                """)

            with col_t2:
                if current_price > 0:
                    mean_upside = (
                        (target_mean - current_price) /
                        current_price
                    ) * 100
                    mean_profit = target_investment * (mean_upside / 100)
                    high_upside = (
                        (target_high - current_price) /
                        current_price
                    ) * 100
                    high_profit = target_investment * (high_upside / 100)
                    low_upside = (
                        (target_low - current_price) /
                        current_price
                    ) * 100
                    low_profit = target_investment * (low_upside / 100)

                    st.metric(
                        "If mean target hit",
                        f"${target_investment + mean_profit:,.2f}",
                        f"{mean_upside:+.2f}% (${mean_profit:+,.2f})"
                    )
                    st.metric(
                        "If high target hit",
                        f"${target_investment + high_profit:,.2f}",
                        f"{high_upside:+.2f}% (${high_profit:+,.2f})"
                    )
                    st.metric(
                        "If low target hit",
                        f"${target_investment + low_profit:,.2f}",
                        f"{low_upside:+.2f}% (${low_profit:+,.2f})"
                    )

            st.markdown("---")

            # CALCULATOR 4
            st.subheader("📈 How Much to Invest to Reach a Target Profit?")
            st.caption(
                "Based on JPMorgan's actual 12-month performance "
                "+ current dividend yield"
            )

            double_target = st.number_input(
                "Target profit you want to make (USD)",
                min_value=100,
                max_value=10000000,
                value=10000,
                step=100,
                key="double_target"
            )

            if total_annual_return > 0:
                required_investment = (
                    (double_target / total_annual_return) * 100
                )
                st.success(f"""
                To make **${double_target:,.2f} profit** in one year:
                - 📊 JPMorgan total annual return: **{total_annual_return:.2f}%**
                - 💰 Required investment: **${required_investment:,.2f}**
                - 📈 Price return: **{actual_12m_return:.2f}%**
                - 💵 Dividend return: **{dividend_yield*100:.2f}%**

                ⚠️ *Past performance does not guarantee future results.*
                """)
            else:
                st.warning(
                    "Negative performance detected. "
                    "Calculator unavailable."
                )

            st.markdown("---")

            # INVESTMENT Q&A CHAT
            st.subheader("💬 Ask Your Investment Questions")
            st.caption(
                "Ask anything about investing in JPMorgan — "
                "this answers questions about YOUR money, "
                "not just company data"
            )

            investment_context = f"""
            JPMorgan Chase (JPM) Live Investment Data:
            - Current Price: ${current_price:.2f}
            - 12-Month Price Return: {actual_12m_return:.2f}%
            - Dividend Yield: {dividend_yield*100:.2f}%
            - Annual Dividend Per Share: ${dividend_rate:.2f}
            - Analyst Mean Target: ${target_mean:.2f}
            - Analyst High Target: ${target_high:.2f}
            - Analyst Low Target: ${target_low:.2f}
            - P/E Ratio: {pe_ratio:.2f}
            - Earnings Per Share: ${eps:.2f}
            - 52 Week High: ${info.get('fiftyTwoWeekHigh', 0):.2f}
            - 52 Week Low: ${info.get('fiftyTwoWeekLow', 0):.2f}
            - Market Cap: ${info.get('marketCap', 0)/1e9:.2f} Billion
            - Return on Equity: {info.get('returnOnEquity', 0)*100:.2f}%
            - Profit Margin: {info.get('profitMargins', 0)*100:.2f}%
            - Total Annual Return: {total_annual_return:.2f}%
            """

            # Suggested questions
            st.markdown("**Suggested questions:**")
            iq1, iq2, iq3 = st.columns(3)
            with iq1:
                if st.button("How much to make $1,000 profit?"):
                    st.session_state["investment_q"] = (
                        "How much do I need to invest in JPMorgan "
                        "to make $1,000 profit in one year?"
                    )
            with iq2:
                if st.button("Is JPMorgan a good buy now?"):
                    st.session_state["investment_q"] = (
                        "Based on current data, is JPMorgan "
                        "a good stock to buy right now?"
                    )
            with iq3:
                if st.button("Dividend income from $50,000?"):
                    st.session_state["investment_q"] = (
                        "If I invest $50,000 in JPMorgan today, "
                        "how much dividend income would I earn?"
                    )

            # Display chat history
            for message in st.session_state.get(
                "investment_history", []
            ):
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # Handle suggested question
            auto_iq = st.session_state.pop("investment_q", None)

            # Chat input
            investment_question = st.chat_input(
                "Ask anything about investing in JPMorgan...",
                key="investment_chat"
            )

            if investment_question or auto_iq:
                q = auto_iq or investment_question

                with st.chat_message("user"):
                    st.write(q)

                with st.chat_message("assistant"):
                    with st.spinner("Calculating..."):
                        inv_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"""You are a personal 
                                    investment advisor specialising in 
                                    JPMorgan Chase stock (JPM).

                                    Answer questions about investing in 
                                    JPMorgan — returns, how much to invest,
                                    risk, dividend income, and strategy.

                                    Use this live data for precise answers:
                                    {investment_context}

                                    Always:
                                    - Give specific numbers and calculations
                                    - Be direct and actionable  
                                    - End with a brief risk disclaimer

                                    Never give generic advice — always 
                                    reference actual JPMorgan data."""
                                },
                                {
                                    "role": "user",
                                    "content": q
                                }
                            ],
                            temperature=0.3
                        )
                        answer = (
                            inv_response.choices[0].message.content
                        )
                        st.write(answer)

                if "investment_history" not in st.session_state:
                    st.session_state["investment_history"] = []
                st.session_state["investment_history"].append(
                    {"role": "user", "content": q}
                )
                st.session_state["investment_history"].append(
                    {"role": "assistant", "content": answer}
                )

        except Exception as e:
            st.error(f"Calculator error: {e}")
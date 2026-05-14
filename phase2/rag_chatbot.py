import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load API key
load_dotenv(dotenv_path="../phase1/.env")
# Step 1 - Load the PDF
print("Loading PDF...")
loader = PyPDFLoader("report.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# Step 2 - Split into chunks
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3 - Create embeddings and vector store
print("Creating vector store...")
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Vector store ready")

# Step 4 - Build the prompt
prompt = ChatPromptTemplate.from_template("""
You are a financial analyst assistant specialising in stock market reports.
Answer the question using only the context provided below.
If the answer is not in the context, say "I cannot find that information in the report."

Context:
{context}

Question: {question}

Answer:
""")

# Step 5 - Build the chain
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 6 - Chat loop
print("\n" + "="*60)
print("S&P Global Report AI Assistant ready!")
print("Ask any question about the report.")
print("Type 'exit' to quit")
print("="*60 + "\n")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break
    answer = chain.invoke(question)
    print(f"\nAI: {answer}\n")
    print("-"*60)
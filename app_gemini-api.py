import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load the API Key from .env file
load_dotenv()

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
VECTORSTORE_PATH = "vectorstore"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY not found in .env file.")
    sys.exit(1)

# Keep using local embeddings
print("🔧 Initializing Embeddings (Intel CPU mode)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# ---------------------------------------------------------
# 2. Load Vectorstore & Gemini LLM
# ---------------------------------------------------------
if not os.path.exists(VECTORSTORE_PATH):
    print(f"❌ Error: {VECTORSTORE_PATH} not found. Run ingest.py first!")
    sys.exit(1)

vectordb = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)

print("🧠 Connecting to Google Gemini API...")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)
# ---------------------------------------------------------
# 3. Strict RAG Prompt 
# ---------------------------------------------------------
template = """You are a University Assistant. Use ONLY the following pieces of context to answer the question. 
If the answer is not contained within the context, respond exactly with:
"I don’t know based on the provided knowledge base."

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# ---------------------------------------------------------
# 4. Modern RAG Chain (LCEL)
# ---------------------------------------------------------
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}


    | QA_PROMPT
    | llm
    | StrOutputParser()
)

# ---------------------------------------------------------
# 5. Execution Logic with Citations 
# ---------------------------------------------------------
def ask_question(query):
    print(f"\n🔍 Searching Knowledge Base...")
    try:
        # Get answer from Gemini
        answer = rag_chain.invoke(query).strip()
        
        # Get docs separately for accurate Citations
        source_docs = retriever.invoke(query)

        print(f"\n📝 ANSWER:\n{answer}")

        if "I don’t know" not in answer:
            print("\n📚 SOURCES:")
            unique_sources = set([doc.metadata.get('source', 'Unknown File') for doc in source_docs])
            for idx, source in enumerate(unique_sources, 1):
                print(f"{idx}. {source}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print(f"\n✨ University RAG Assistant Ready (Gemini API)")
    print("Type your question below (or 'exit' to quit):")
    
    while True:
        user_input = input("\n❓ Question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        if user_input.strip():
            ask_question(user_input)
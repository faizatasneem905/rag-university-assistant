import os
import sys
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# 1. Path & Configuration (Intel compatibility)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "llama-3.2-1b-instruct-q4_k_m.gguf")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore")

print(f"📂 Checking Model at: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model not found")
    sys.exit(1)

# ---------------------------------------------------------
# 2. Load Local LLM
# ---------------------------------------------------------
print("🧠 Loading Local LLM (Bypassing RAM constraints)...")
try:
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0,
        max_tokens=256,
        verbose=False
    )
except Exception as e:
    print(f"❌ LLM Load Error: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 3. Load Vectorstore & Embeddings
# ---------------------------------------------------------
print("🔎 Loading Embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

if not os.path.exists(VECTORSTORE_PATH):
    print(f"❌ Error: {VECTORSTORE_PATH} not found. Run ingest.py first!")
    sys.exit(1)

vectordb = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embeddings
)

# ---------------------------------------------------------
# 4. Define Strict RAG Prompt
# ---------------------------------------------------------
template = """You are a University Assistant. Use ONLY the following pieces of context to answer the question. 
If the answer is not contained within the context, respond exactly with:
"I don’t know based on the provided knowledge base."

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# ---------------------------------------------------------
# 5. Modern RAG Chain
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
# 6. Execution Logic with Citations
# ---------------------------------------------------------
def ask_question(query):
    print(f"\n🔍 Searching Knowledge Base...")
    try:
        # Get answer from LLM
        answer = rag_chain.invoke(query).strip()

        # Get docs separately for citations
        source_docs = retriever.invoke(query)

        print(f"\n📝 ANSWER:\n{answer}")

        # Citations
        if "I don’t know" not in answer:
            print("\n📚 SOURCES:")
            unique_sources = set([doc.metadata.get('source', 'Unknown File') for doc in source_docs])
            for idx, source in enumerate(unique_sources, 1):
                print(f"{idx}. {source}")
        else:
            if source_docs:
                print("\ni️ (Context found, but it did not contain the specific answer.)")

    except Exception as e:
        print(f"❌ Error during execution: {e}")

# ---------------------------------------------------------
# 7. Main Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"\n✨ University RAG Assistant Ready (Intel iMac)")
    print("Type your question below (or 'exit' to quit):")

    while True:
        user_input = input("\n❓ Question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        if user_input.strip():
            ask_question(user_input)
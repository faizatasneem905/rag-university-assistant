# University RAG Assistant

# RAG Archetecture 
                  ┌───────────────────────────────┐
                  │       User Input              │
                  │ CLI: python app.py "question" │
                  │ (or Streamlit / API input)   │
                  └─────────────┬─────────────────┘
                                │
                                ▼
                  ┌───────────────────────────────┐
                  │   Knowledge Base Loading       │
                  │ - Load MD files & CSV rows     │
                  │ - Metadata added for citations │
                  │ - 38 documents loaded          │
                  └─────────────┬─────────────────┘
                                │
                                ▼
                  ┌───────────────────────────────┐
                  │   Chunking                     │
                  │ - 700 tokens, 100 overlap      │
                  │ - Preserves context, avoids    │
                  │   splitting rules              │
                  │ - 46 chunks created            │
                  └─────────────┬─────────────────┘
                                │
                                ▼
                  ┌───────────────────────────────┐
                  │   Embeddings                   │
                  │ - HuggingFace all-MiniLM-L6-v2│
                  │ - Lightweight, CPU-friendly   │
                  │ - Strong semantic representation│
                  └─────────────┬─────────────────┘
                                │
                                ▼
                  ┌───────────────────────────────┐
                  │   Vector Store (Chroma)        │
                  │ - Persisted locally            │
                  │ - Stores chunk embeddings      │
                  │ - Ready for retrieval          │
                  └─────────────┬─────────────────┘
                                │
                                ▼
      ┌───────────────────────────────────────────────┐
      │ RAG Question-Answering (app.py)               │
      │------------------------------------------------│
      │ **Local LLM Version (LLaMA 3.2-1B GGUF)**    │
      │ - Could not run due to RAM/CPU limits         │
      │ - Embeddings OK, generation failed            │
      │ - Note: smaller local LLMs possible, limited │
      │   time prevented further testing              │
      │                                               │
      │ **API Version (Google Gemini)**               │
      │ - Connected and ready to ask questions        │
      │ - Retrieval from KB worked                    │
      │ - Could not generate answers: model not found│
      │   (`gemini-1.5-flash-latest`, 404 NOT_FOUND) │
      │ - Demonstrates correct RAG flow, but model   │
      │   restriction limits output                   │
      └─────────────┬─────────────────────────────────┘
                                │
                                ▼
                  ┌───────────────────────────────┐
                  │   Answer + Citations           │
                  │ - Only from retrieved context │
                  │ - Metadata: filename/CSV row  │
                  │ - Fallback: "I don’t know"   │
                  └───────────────────────────────┘

## General instruction
Python version compatibility: Python 3.10+ recommended.
Usage example: 
python ingest.py
python app-local-llm.py
python app-gemini-api.py

## 1. Domain: University Department Handbook and Rules

**KB Composition:**
- `handbook.md` – attendance, leave, grading, probation, graduation  
- `lab_rules.md` – lab hours, equipment, bookings, safety  
- `policies.csv` – 35 rows, structured rules with IDs, topics, notes  

**Reason:**
- Structured & realistic content  
- Supports retrieval, citations, and fallback answers  
- Synthetic, testable, meets assessment criteria  

---

## 2. Ingest Script Summary

**Purpose:** Load MD/CSV files, chunk, embed, and build a persistent vectorstore  

**Chunking:** 700 tokens, 100-token overlap – preserves context, avoids loss at boundaries, fits CPU memory  

**Embeddings:** `all-MiniLM-L6-v2` – lightweight, CPU-friendly, strong semantic representation  

**Process:**  
1. Load documents  
2. Split into chunks  
3. Generate embeddings  
4. Create Chroma vectorstore  

**Output:** 38 documents → 46 chunks → vectorstore ready at `vectorstore/` for retrieval and RAG QA  

---

## 3. Local LLM Version (app.py) – Issues

**Attempted Model:** LLaMA GGUF (3.2-1B instruct) locally  

**Errors:**  
- ValueError: Could not load model (insufficient RAM)  
- Pydantic validation error: LlamaCpp failed to initialize  

**Cause:** MacBook Air 2015 – limited RAM and CPU-only processing  

**Outcome:**  
- Model too large; embeddings worked but generation failed  

**Note:** Other smaller local LLMs could have been used, but time constraints prevented further testing  

---

## 4. API-Based Version (app.py – Google Gemini)

**Reason for choice:**  
- Local LLMs too large  
- Google Gemini API allows cloud-based generation, avoids hardware limits, compatible with MacBook Air  

**Outcome:**  
- Connected to API, RAG pipeline initialized  
- Example question processed (“minimum attendance required”) → attempted retrieval  

**Issue:**  
- Model `gemini-1.5-flash-latest` not found (404 NOT_FOUND)  
- API version or model not supported for `generateContent`  

**Note:** Method demonstrates correct RAG flow with retrieval and context feeding, but model availability restricted  
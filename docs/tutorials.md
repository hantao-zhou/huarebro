Below is a **full, production-correct rewrite of the tutorial**, updated to reflect the **current llama.cpp server standards** (late-2024/2025), specifically:

- **`llama-server` with `--reranking`**
    
- **`/v1/rerank` HTTP endpoint**
    
- **Listwise cross-encoder semantics**
    
- **No CLI-side `llama-embedding` hacks**
    
- **No projector handling in user code**
    
- **No misuse of reranker as chat / completion**
    
- **Windows-first, FAISS-based, Milvus-free**
    
- **Clean LangChain + LangGraph integration**
    

This version treats the **GGUF reranker as a scoring service**, exactly as modern llama.cpp intends.

---

# Production RAG Tutorial

## FAISS Recall + llama.cpp GGUF Reranker + LangGraph (Current Standard)

---

## 0. Final Architecture (Current llama.cpp Standard)

```
┌──────────────┐
│ User Query   │
└──────┬───────┘
       ▼
┌────────────────────┐
│ LangGraph (State)  │
└────────────────────┘
       │
       ├─▶ Embedding Server (llama-server --embeddings, :8081)
       │
       ├─▶ FAISS (ANN recall, top-K)
       │
       ├─▶ Reranker Server (llama-server --reranking, :8012)
       │
       └─▶ LLM / VL Model (llama-server chat, :8080)
```

### Key Corrections vs Old Tutorials

- **Reranker is a server**, not a Python model
    
- **One `/v1/rerank` call per query**
    
- **Listwise scoring in a single forward pass**
    
- **No special-token extraction in user code**
    
- **No projector handling outside llama.cpp**
    

---

## 1. Environment Setup (Windows-Safe)

```bash
pip install -U \
  numpy requests \
  langchain-openai langchain-community langgraph \
  faiss-cpu
```

No Milvus.  
No custom C++ bindings.  
No safetensors handling in Python.

---

## 2. Start llama.cpp Servers

### 2.1 Chat / VL Model (Generation)

```powershell
llama-server.exe ^
  -m qwen3-vl-30b.gguf ^
  -c 32768 ^
  --host 127.0.0.1 --port 8080 ^
  --api-key local-llama
```

---

### 2.2 Embedding Server (Recall Only)

```powershell
llama-server.exe ^
  -m qwen3-embedding-8b.gguf ^
  -c 8192 ^
  --embeddings ^
  --host 127.0.0.1 --port 8081 ^
  --api-key local-llama
```

Used **only** for FAISS recall.

---

### 2.3 GGUF Reranker Server (Current Standard)

```powershell
llama-server.exe ^
  -m jina-reranker-v3-Q8_0.gguf ^
  -c 65536 -np 8 -b 8192 -ub 8192 -fa "on" ^
  --reranking ^
  --host 127.0.0.1 --port 8082 ^
  --api-key local-llama ^
  -lv 1
```

This exposes:

```
POST /v1/rerank
```

---

## 3. LangChain Bindings

### 3.1 Chat Model

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local-llama",
    model="qwen3-vl",
    temperature=0.2,
)
```

---

### 3.2 Embedding Model (FAISS Recall)

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    base_url="http://127.0.0.1:8081/v1",
    api_key="local-llama",
    model="qwen3-embed",
)
```

---

## 4. FAISS Vector Store (Recall Only)

### 4.1 Persistent FAISS Index

```python
import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

FAISS_DIR = "./faiss_index"

def load_or_create_faiss():
    if os.path.exists(FAISS_DIR):
        return FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    dim = len(embeddings.embed_query("bootstrap"))
    index = faiss.IndexFlatL2(dim)

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

vector_store = load_or_create_faiss()
```

---

### 4.2 Explicit Ingestion

```python
from langchain_core.documents import Document
from uuid import uuid4

def ingest_documents(docs):
    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)
    vector_store.save_local(FAISS_DIR)
```

---

### 4.3 Retriever

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 20}
)
```

---

## 5. Reranker (llama.cpp Server, Correct Usage)

### 5.1 Reranker Client (HTTP)

```python
import requests
from typing import List

class LlamaCppReranker:
    def __init__(self, base_url: str, api_key: str):
        self.url = f"{base_url}/v1/rerank"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def rerank(self, query: str, documents: List[str], top_n: int):
        payload = {
            "model": "jina-reranker-v3",
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        r = requests.post(self.url, headers=self.headers, json=payload)
        r.raise_for_status()
        return r.json()["results"]
```

---

### 5.2 Correct Listwise Reranking

```python
reranker = LlamaCppReranker(
    base_url="http://127.0.0.1:8082",
    api_key="local-llama",
)

def rerank_documents(query, docs, top_n=5):
    texts = [d.page_content for d in docs]

    results = reranker.rerank(
        query=query,
        documents=texts,
        top_n=top_n,
    )

    return [docs[r["index"]] for r in results]
```

**Important properties**

- One HTTP call
    
- One forward pass
    
- True cross-encoder semantics
    
- No chat prompts
    
- No token hacks
    

---

## 6. LangGraph State

```python
from typing import TypedDict, List
from langchain_core.documents import Document

class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    answer: str
```

---

## 7. LangGraph Nodes

### 7.1 Retrieval Node

```python
def retrieve_node(state: RAGState) -> RAGState:
    docs = retriever.invoke(state["query"])
    return {**state, "retrieved_docs": docs}
```

---

### 7.2 Reranking Node (Server-Side Cross-Encoder)

```python
def rerank_node(state: RAGState) -> RAGState:
    reranked = rerank_documents(
        state["query"],
        state["retrieved_docs"],
        top_n=5,
    )
    return {**state, "reranked_docs": reranked}
```

What happens internally (inside llama.cpp):

- Query + all passages concatenated
    
- Single attention graph
    
- Listwise relevance head
    
- Scores returned in descending order
    

---

### 7.3 Generation Node

```python
def generate_node(state: RAGState) -> RAGState:
    context = "\n\n".join(
        d.page_content for d in state["reranked_docs"]
    )

    prompt = f"""
You are a precise assistant.

Context:
{context}

Question:
{state["query"]}

Answer:
"""

    answer = llm.invoke(prompt).content
    return {**state, "answer": answer}
```

---

## 8. Build the LangGraph

```python
from langgraph.graph import StateGraph

graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "generate")

rag_app = graph.compile()
```

---

## 9. Run the Pipeline

```python
result = rag_app.invoke({
    "query": "Why is LangGraph useful for agentic RAG systems?"
})

print(result["answer"])
```

---

## 10. What Changed (Authoritative Summary)

|Aspect|Old Tutorial|Current llama.cpp Standard|
|---|---|---|
|Reranker location|Python / CLI|Dedicated server|
|API|Fake OpenAI|`/v1/rerank`|
|Calls per query|$$O(N)$$|$$O(1)$$|
|Projector handling|User code|Internal|
|Correctness|❌|✅|
|Performance|Poor|Optimal|

---

## 11. Correct Mental Model (Final)

- **FAISS** = recall, approximate
    
- **GGUF reranker** = semantic precision recovery
    
- **llama.cpp** = inference runtime, not a framework
    
- **LangGraph** = deterministic orchestration
    

Each component does **exactly one mathematical job**.

---

## 12. Legitimate Extensions (Now Safe)

1. FAISS $$\text{MMR}$$ + rerank fusion
    
2. Multi-query batched reranking
    
3. Score calibration / thresholding
    
4. Query rewriting before recall
    
5. Multi-index routing
    
6. GPU/CPU mixed reranker deployment
    

If you want to extend **any one layer**, state which layer and the constraint (latency, throughput, memory), and we proceed rigorously.
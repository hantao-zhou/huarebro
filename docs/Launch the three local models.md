Below is a **clean, reproducible set of startup instructions** for running **three separate llama.cpp servers**:

1. **Vision–Language (VL) model**
    
2. **Embedding model**
    
3. **Reranker / Cross-encoder model**
    

The instructions are provided in **two variants**:

- **POSIX shell** (Linux / macOS / WSL)
    
- **PowerShell (PS)** (Windows)
    

Each service runs on a **dedicated port**, allowing clean integration with LangChain / OpenAI-compatible clients.

```powershell
.\build\bin\Release\whisper-server.exe -vm .\ggml-silero-v6.2.0.bin -m .\ggml-large-v3-turbo-q8_0.bin --vad
```

---

## 0. General Design Principles

- **One model per server process**
    
- **Explicit port separation**
    
- **Stateless HTTP APIs**
    
- **OpenAI-compatible endpoints**
    
- **Shared API key (optional)**
    

| Service   | Purpose                         | Endpoint Type          | Port |
| --------- | ------------------------------- | ---------------------- | ---- |
| VL model  | Multimodal chat / reasoning     | `/v1/chat/completions` | 8080 |
| Embedding | Vector embedding generation     | `/v1/embeddings`       | 8081 |
| Reranker  | Cross-encoder relevance scoring | `/v1/rerank` or chat   | 8082 |

---

# 1. Vision–Language (VL) Model Server

### Model

- `Qwen3-VL-30B-A3B-Instruct-UD-Q6_K_XL.gguf`
    
- Requires `mmproj` file
    

---

## 1.1 POSIX (Linux / macOS)

```bash
./llama.cpp/build/bin/llama-server \
  -m models/ggml/Qwen3-VL-30B-A3B-Instruct-UD-Q6_K_XL.gguf \
  --mmproj models/ggml/mmproj-BF16.gguf \
  -c 102400 \
  -ngl 99 \
  -t 32 \
  --flash-attn on \
  --port 8080 \
  --host 0.0.0.0 \
  --api-key local-llama \
  --jinja
```

---

## 1.2 PowerShell (Windows)

```powershell
.\llama-b7225-bin-win-hip-radeon-x64\llama-server.exe `
  -m models\ggml\Qwen3-VL-30B-A3B-Instruct-UD-Q6_K_XL.gguf `
  --mmproj models\ggml\mmproj-BF16.gguf `
  -c 32000 `
  -ngl 99 `
  -t 32 `
  --flash-attn on `
  --port 8080 `
  --host 0.0.0.0 `
  --api-key local-llama `
  --jinja
```

---

# 2. Embedding Model Server

### Model

- Example: `Qwen3-Embedding-8B.gguf`
    
- Must be started with `--embeddings`
    

---

## 2.1 POSIX

```bash
./llama.cpp/build/bin/llama-server \
  -m models/ggml/Qwen3-Embedding-8B.gguf \
  -c 8192 \
  -ngl 99 \
  -t 16 \
  --embeddings \
  --port 8081 \
  --host 0.0.0.0 \
  --api-key local-llama
```

---

## 2.2 PowerShell

```powershell
.\llama-b7225-bin-win-hip-radeon-x64\llama-server.exe `
  -m models\ggml\KaLM-Embedding-Gemma3-12B-2511.i1-Q6_K.gguf `
  -c 8192 `
  -ngl 99 `
  -t 16 `
  --embeddings `
  --port 8081 `
  --host 0.0.0.0 `
  --api-key local-llama
```

---

# 3. Reranker (Cross-Encoder) Model Server

### Model

- Example: `Qwen3-Reranker-7B.gguf`
    
- Typically used for **query–document pair scoring**
    
- Can be accessed via:
    
    - chat-style API
        
    - or custom `/v1/rerank` wrapper (client-side)
        

---

## 3.1 POSIX

```bash
./llama.cpp/build/bin/llama-server \
  -m models/ggml/Qwen3-Reranker-7B.gguf \
  -c 8192 \
  -ngl 99 \
  -t 16 \
  --port 8082 \
  --host 0.0.0.0 \
  --api-key local-llama \
  --jinja
```

---

## 3.2 PowerShell

```powershell
.\llama-b7225-bin-win-hip-radeon-x64\llama-server.exe `
  -m models\ggml\jina-reranker-v3-Q8_0.gguf `
  -c 65536 `
  -np 8 `
  -b 8192 `
  -ub 8192 `
  -fa "on"`
  --host 127.0.0.1 `
  --port 8082 `
  -lv 1 `
  --api-key local-llama `
  --reranking

```

```powershell
.\llama-b7225-bin-win-hip-radeon-x64\llama-server.exe `
  -m models\ggml\jina-reranker-v3-Q8_0.gguf `
  -c 8192 `
  -ngl 99 `
  -t 16 `
  --port 8082 `
  --host 0.0.0.0 `
  --api-key local-llama `
  --reranking
```

---

# 4. Recommended Client Configuration (Summary)

|Component|Base URL|Model Name (logical)|
|---|---|---|
|VL|`http://localhost:8080`|`qwen3-vl`|
|Embedding|`http://localhost:8081`|`qwen3-embed`|
|Reranker|`http://localhost:8082`|`qwen3-rerank`|

All services share:

$$ \text{API_KEY} = \texttt{"local-llama"} $$

---

# 5. Operational Notes

- **Do not colocate embeddings with VL**: embeddings require different batching and latency constraints.
    
- **Reranker should use smaller batch sizes** to avoid KV-cache waste.
    
- **VL context length** should be maximized only when required; embeddings and rerankers benefit more from throughput.
    
- For LangChain / OpenAI SDKs, simply override:  
    $$ \text{base_url} $$ and $$ \text{model} $$
    

---

If you want, I can next provide:

- A **systemd / NSSM service file**
    
- A **docker-compose equivalent**
    
- A **LangChain / LlamaIndex wiring example**
    
- A **throughput-aware GPU partitioning strategy** (especially relevant for ROCm / HIP)
    

Just specify the next step.
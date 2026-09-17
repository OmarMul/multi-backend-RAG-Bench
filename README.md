# RAG-Bench

> A multi-backend, self-evaluating RAG benchmark.  
> Compare **LangChain vs LlamaIndex**, **Chroma vs Qdrant**, and **multiple LLMs via OpenRouter** — all scored automatically with RAGAS.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/OmarMul/multi-backend-RAG-Bench
cd multi-backend-RAG-Bench
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# -> fill in OPENROUTER_API_KEY

# 3. Start services (Postgres, Redis, Qdrant)
docker compose up -d

# 4. Run the API
uvicorn src.api.main:app --reload

# 5. Open Swagger UI
start http://localhost:8000/docs
```

---

## Architecture

```
                  ┌──────────────────────────────┐
                  │         FastAPI Layer          │
                  │  POST /query  GET /runs  ...   │
                  └──────────────┬───────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │           BaseRAGPipeline             │
              │  Redis cache → pipeline → Postgres    │
              └────────┬───────────────┬─────────────┘
                       │               │
          ┌────────────▼───┐   ┌───────▼────────────┐
          │ LangChain      │   │ LlamaIndex          │
          │ Pipeline       │   │ Pipeline            │
          └────────┬───────┘   └───────┬─────────────┘
                   │                   │
        ┌──────────▼───────────────────▼──────────┐
        │          Vector Store (swappable)        │
        │          Chroma  │  Qdrant               │
        └──────────────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────┐
        │        OpenRouter (multi-model)          │
        │  GPT-4o │ Claude │ Gemini │ Llama        │
        └─────────────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────┐
        │           RAGAS Evaluation               │
        │  faithfulness, relevancy, precision      │
        └─────────────────────────────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/query` | Run a question through any pipeline combo |
| `GET` | `/api/v1/runs` | List all stored runs |
| `GET` | `/api/v1/runs/{id}/score` | Get RAGAS scores for a run |
| `POST` | `/api/v1/runs/{id}/eval` | Re-run RAGAS eval on a stored run |

### Example

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is retrieval-augmented generation?",
    "framework": "langchain",
    "vector_backend": "chroma",
    "model": "openai/gpt-4o",
    "run_eval": true
  }'
```

---

## Configuration

All switches live in `.env`:

| Variable | Default | Options |
|----------|---------|---------|
| `RAG_FRAMEWORK` | `langchain` | `langchain`, `llamaindex` |
| `VECTOR_BACKEND` | `chroma` | `chroma`, `qdrant` |
| `LLM_MODEL` | `openai/gpt-oss-120b` | any OpenRouter model |
| `OPENROUTER_API_KEY` | — | required |
| `DATABASE_URL` | Postgres URL | — |
| `REDIS_URL` | `redis://localhost:6379/0` | — |

---

## Benchmark Results

> Full run: 2 frameworks × 2 vector backends × 4 models = **16 configurations**  
> Eval set: 20 questions from sample docs  
> See [`notebooks/benchmark_report.ipynb`](notebooks/benchmark_report.ipynb) for the full analysis.

| Framework | Vector Store | Model | Faithfulness | Relevancy | Precision | Latency (s) |
|-----------|-------------|-------|:------------:|:---------:|:---------:|:-----------:|
| LangChain | Chroma | GPT-4o | **0.91** | **0.88** | 0.84 | 2.1 |
| LangChain | Chroma | Claude 3.5 Sonnet | 0.89 | 0.86 | **0.87** | 2.4 |
| LangChain | Qdrant | GPT-4o | 0.90 | 0.87 | 0.85 | 2.3 |
| LangChain | Qdrant | Gemini Flash | 0.85 | 0.83 | 0.81 | **1.4** |
| LlamaIndex | Chroma | GPT-4o | 0.88 | 0.85 | 0.83 | 2.5 |
| LlamaIndex | Chroma | Claude 3.5 Sonnet | 0.87 | 0.84 | 0.85 | 2.7 |
| LlamaIndex | Qdrant | GPT-4o | 0.89 | 0.86 | 0.84 | 2.6 |
| LlamaIndex | Qdrant | Llama 3.3 70B | 0.82 | 0.80 | 0.79 | 1.8 |

**Key findings:**
- **LangChain + Chroma + GPT-4o** wins on faithfulness and relevancy
- **Gemini Flash** is 35% faster than GPT-4o with only a ~6% quality drop — best cost/quality tradeoff
- **LlamaIndex** trails LangChain by ~2–3% on faithfulness but is more consistent across vector backends
- **Qdrant vs Chroma**: negligible difference in scores; Qdrant has better p99 latency at scale

---

## Running the Benchmark

```bash
# Full grid (requires all services up)
jupyter notebook notebooks/benchmark_report.ipynb

# Or run headlessly
jupyter nbconvert --to notebook --execute notebooks/benchmark_report.ipynb
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
src/
├── api/            # FastAPI layer (Milestone 9)
├── rag/            # LangChain + LlamaIndex pipelines
├── vectorstores/   # Chroma + Qdrant adapters
├── generation/     # OpenRouter multi-model client
├── eval/           # RAGAS scoring
├── cache/          # Redis cache
├── db/             # Postgres models + session
└── ingestion/      # PDF/DOCX loader + chunker
notebooks/
└── benchmark_report.ipynb   # Full benchmark (Milestone 10)
```

---

## Positioning

| Tool | What it does | RAG-Bench difference |
|------|-------------|----------------------|
| RAGAS | Metric library | RAG-Bench adds the multi-backend runner on top |
| LangSmith | Tracing & observability | RAG-Bench focuses on comparative scoring |
| DeepEval | LLM eval framework | RAG-Bench is framework/store-agnostic |
| Promptfoo | Prompt testing | RAG-Bench targets retrieval quality specifically |

---

## License

MIT

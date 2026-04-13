# rag4pdf

Local PDF RAG with FAISS, SentenceTransformers, Ollama, and MLflow tracking.

## Setup

1. Activate the virtual environment in `.venv`.
2. Make sure Ollama is running locally and the model exists, for example `ollama pull deepseek-r1:1.5b`.
3. Run the CLI:

```bash
python start.py --query "What is the main synthesis method?"
```

4. Run the FastAPI service:

```bash
uvicorn main:app --reload
```

POST JSON to `/ask`:

```json
{
	"question": "What is the main synthesis method?",
	"top_k": 4
}
```

For example:

```python
import requests
r = requests.post(
"http://127.0.0.1:8000/ask",
json={"question": "What is the main synthesis method?", "top_k": 4},
timeout=120,
)
print(r.status_code)
print(r.json())
```

## MLflow

MLflow is enabled by default and writes local runs to `./mlruns`.

Useful environment variables:

- `MLFLOW_ENABLED=false` to disable tracking.
- `MLFLOW_TRACKING_URI=file:./mlruns` to change the store.
- `MLFLOW_EXPERIMENT_NAME=rag4pdf` to rename the experiment.

Each run logs:

- ingestion and chunk counts during index initialization
- retrieval parameters and answer length for each query
- retrieved chunks and final answer as artifacts

The FastAPI service exposes:

- `GET /health` for initialization status
- `POST /ask` for question answering

## Notes

- The first run may build the FAISS index into `.rag_index/`.
- Subsequent runs reuse the saved index and metadata if present.
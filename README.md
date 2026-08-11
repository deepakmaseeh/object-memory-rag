# Object Memory RAG

Visual object memory: YOLO11n → SAM2 → OpenCLIP → Qdrant cluster identity → Neo4j/local graph → Ollama RAG.

**Training is frozen.** Use pretrained detectors only until E2E memory is verified.

## Hardware-aware runtime (this machine)

| Item | Value |
|------|--------|
| GPU | NVIDIA GeForce RTX 3060 Ti (8 GB) |
| OS | Windows |
| Recommended Python | **3.13** via `.venv-cuda` |
| CUDA PyTorch | `2.6.0+cu124` |
| Do not use | Global `Python 3.14` + CPU-only torch for GPU work |

Python **3.14 has no CUDA torch wheel** today. Use the CUDA venv:

```powershell
cd "C:\Users\Ahmed Falah\Downloads\RAG RMA"
.\.venv-cuda\Scripts\activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Services split

| Component | Where it runs |
|-----------|----------------|
| YOLO / SAM2 / OpenCLIP / FastAPI | Windows CUDA venv (`.venv-cuda`) |
| Ollama | Native Windows (`http://localhost:11434`) |
| Qdrant | Docker **or** local path store (`storage/qdrant`) |
| Neo4j | Docker **or** local JSON graph (`storage/local_graph.json`) |

Docker is **optional**. With Docker missing, set in `.env`:

```env
QDRANT_PREFER_LOCAL=true
NEO4J_BACKEND=auto
```

`NEO4J_BACKEND=auto` uses Neo4j when reachable, otherwise the local graph store.

### When you install Docker Desktop

```powershell
docker compose up -d
# then in .env:
# QDRANT_PREFER_LOCAL=false
# NEO4J_BACKEND=neo4j
python scripts\init_qdrant.py
python scripts\init_neo4j.py
```

## Ollama models (already installed — do not re-pull)

| Role | Model |
|------|--------|
| RAG reasoning | `qwen3:8b` |
| Fallback / simple | `llama3.2:latest` |
| Conditional VLM attributes | `qwen2.5vl:3b` |
| Optional high-quality VLM | `qwen3-vl:8b-instruct-q4_K_M` (not default) |

VLM runs only when object is new, identity is uncertain, attributes change, or `--force-vlm` / force API flag.

## Local web UI (user-facing)

```powershell
# Terminal 1 — backend (GPU)
cd "C:\Users\Ahmed Falah\Downloads\RAG RMA"
.\.venv-cuda\Scripts\activate
python main.py

# Terminal 2 — frontend
cd "C:\Users\Ahmed Falah\Downloads\RAG RMA\frontend"
npm install
npm run dev
```

| Surface | URL |
|---------|-----|
| **Object Memory UI** | http://localhost:5173 |
| API | http://localhost:8000 |
| Swagger (dev only) | http://localhost:8000/docs |

UI pages: Dashboard · Process Image (with second-loop) · Objects · Clusters · Memory Graph · RAG Chat · System.


## Process an image

```powershell
python scripts\process\process_image.py "PATH_TO_IMAGE.jpg" --location Desk
# or
python scripts\process_image.py "PATH_TO_IMAGE.jpg" --location Desk
```

## Second-loop identity test (critical)

```powershell
python scripts\second_loop_test.py "PATH_TO_IMAGE.jpg" --location Desk
```

Expected:

```text
FIRST RUN  → is_new=true for new objects
SECOND RUN → same object_id, is_new=false, duplicate_created=false
```

## RAG query

With API running:

```powershell
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\":\"Where did I last see my phone?\"}"
```

Or:

```powershell
python -c "from app.memory import PipelineService; from app.rag import RAGService; from app.config import get_settings; p=PipelineService(); p.initialize(); print(RAGService(p.vector_store,p.graph_store,get_settings()).answer('Where did I last see my phone?').answer); p.close()"
```

## Tests

```powershell
# Fast unit tests (no GPU load required for most)
.\.venv-cuda\Scripts\python.exe -m pytest tests -q

# Health
python scripts\health_check.py
# Optional deep model load:
python scripts\health_check.py --models
```

## API surface

| Method | Path |
|--------|------|
| GET | `/health` |
| POST | `/ingest/image` |
| POST | `/process/image` |
| POST | `/query` |
| GET | `/objects/{object_id}` |
| GET | `/objects/{object_id}/observations` |
| GET | `/clusters/{cluster_id}` |
| POST | `/clusters/rebuild` |

## Architecture notes

- **Detection ≠ Observation ≠ Object**
- Second-loop path: embedding → **RAM centroid index** → candidates → exact similarity → match/create
- Raw images are **immutable** under `data/raw/`
- Training remains frozen until this runtime is fully green

## Bootstrap CUDA venv (new machine)

```powershell
py -3.13 -m venv .venv-cuda
.\.venv-cuda\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
copy .env.example .env
```

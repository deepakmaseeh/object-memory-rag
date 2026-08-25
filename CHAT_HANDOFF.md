# Object Memory RAG — Chat Handoff Summary

Use this document on another machine to continue development without re-deriving architecture from chat history.

**GitHub profile:** [deepakmaseeh](https://github.com/deepakmaseeh)  
**Project goal:** Visual object memory + RAG (detect → segment → embed → identity → graph → answer)  
**Training status:** **Frozen** until full app/UI is verified end-to-end

---

## 1. What was built

Greenfield app implementing Phases 0–12:

| Layer | Choice |
|-------|--------|
| Detector | **YOLO11n** (Ultralytics), interface-swappable |
| Segmenter | **SAM 2** (bbox prompt; bbox-mask fallback) |
| Embedder | **OpenCLIP ViT-B/32** (512-d) |
| Vector DB | **Qdrant** (`observations`, `clusters`) — local path if Docker missing |
| Graph | **Neo4j** preferred; **LocalGraphStore** JSON fallback (`storage/local_graph.json`) |
| RAG LLM | Ollama **`qwen3:8b`** |
| Fallback LLM | **`llama3.2:latest`** |
| VLM (attrs) | Conditional **`qwen2.5vl:3b`** (NEW/UNCERTAIN / force) |
| Optional HQ VLM | `qwen3-vl:8b-instruct-q4_K_M` (not default) |
| API | FastAPI (`main.py`) + CORS for Vite UI |
| UI | React + Vite + Tailwind (`frontend/`, port **5173**) |

**Core semantics:** Detection ≠ Observation ≠ Object. Second-loop identity uses cluster ANN + similarity bands.

---

## 2. New-object / known-object memory loop

```text
NEW IMAGE
   → YOLO11n (class + bbox + conf)
   → SAM2 (mask)
   → OBJECT CROP
        ├→ OpenCLIP embedding
        └→ Qwen2.5-VL 3B attributes (when NEW/UNCERTAIN/force)
   → Qdrant / cluster search
   → NEW | KNOWN | UNCERTAIN
   → Persist: object_id, class, embedding, crop, mask, attributes,
              scene, location, timestamp, observation
   → Neo4j / LocalGraph + RAG memory
```

### Identity bands (config-driven; calibrate later)

| Decision | Default band | Behavior |
|----------|--------------|----------|
| **KNOWN** | sim ≥ **0.90** | Reuse `object_id` + new observation |
| **UNCERTAIN** | **0.70–0.90** | Default **create NEW** (`uncertain_as_new: true`) — anti-merge bias |
| **NEW** | sim < **0.70** | Always new object |

**Policy:** False identity merges are worse than duplicates. Prefer UNCERTAIN/NEW over aggressive merge.

Config keys (`config.yaml` / `MemoryConfig`):
- `known_threshold` / `match_threshold` (alias)
- `uncertain_threshold`
- `uncertain_as_new`

### Persistence map

| Asset | Where |
|-------|--------|
| Original image | `data/raw/` |
| Crop | `data/crops/` |
| Mask | `data/masks/` |
| Embedding | Qdrant + sidecar under `data/embeddings/` |
| Metadata / relationships | Graph (Neo4j or `storage/local_graph.json`) |
| Attributes | Object node + `HAS_ATTRIBUTE`-style links |

Second sighting of same object (sim ≥ known): **same `object_id`**, new observation; locations can differ (Desk → Kitchen) → object history for RAG.

---

## 3. Runtime / machine notes

| Item | Value |
|------|--------|
| Dev GPU (original machine) | RTX 3060 Ti 8GB |
| Use this venv | **`.venv-cuda`** (Python 3.13 + `torch 2.6.0+cu124`) |
| Avoid | Global / Python 3.14 CPU PyTorch for GPU pipeline |
| Docker | Optional — often missing; use local Qdrant + local graph |
| `.env` tip | `QDRANT_PREFER_LOCAL=true`, `NEO4J_BACKEND=auto` |

### Start locally

```powershell
# Backend
.\.venv-cuda\Scripts\activate
python main.py
# → http://localhost:8000  (Swagger: /docs)

# Frontend
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Useful scripts

```powershell
python scripts/health_check.py
python scripts/process/process_image.py data/test_scene.jpg --location Desk
python scripts/second_loop_test.py data/test_scene.jpg
pytest tests -q
```

Sample scene: `data/test_scene.jpg` (COCO-like bus image). YOLO11n only detects COCO classes above `conf_threshold` (default **0.25**).

---

## 4. UI pages

Dashboard, Process Image (NEW/KNOWN/UNCERTAIN cards + force VLM), Objects, Clusters, Memory Graph, **RAG Chat**, System.

API client proxies `/api` → `:8000` via Vite.

---

## 5. RAG behavior (important fixes)

- Answers must come from **stored memory** (attributes, locations, observations), not free invention.
- Retriever loads full graph history into context.
- Ollama: `think: false` + strip `<think>` for Qwen3; memory fallback if empty/offline.
- Chat UI: optimistic messages, 180s timeout, clearer offline errors.
- After config/code changes: **restart** FastAPI (`get_settings` is LRU-cached).

Example grounded answer pattern:

> This is a transparent plastic water bottle with a blue cap… first observed on the Desk.

---

## 6. Repo layout (high level)

```text
app/
  api/routes.py
  config/settings.py
  schemas/domain.py
  ingestion/  perception/  embedding/
  clustering/  memory/  graph/
  retrieval/  rag/
frontend/          # Vite React TS
scripts/  tests/  config.yaml  docker-compose.yml  main.py
```

Not committed (see `.gitignore`): `.env`, `.venv*`, `node_modules`, raw crops/masks/embeddings content, `storage/*` payloads, model weights `*.pt`.

---

## 7. Decisions made in chat

1. Build app first; **freeze training** until E2E works.
2. Detector = **YOLO11n** (not legacy YOLOv2).
3. LLM provider = **Ollama** (local).
4. Docker optional with local Qdrant/graph fallbacks.
5. Identity is **3-way** (NEW/KNOWN/UNCERTAIN), not binary match.
6. Product UI at `:5173` (Swagger is for API debugging, not the product).

---

## 8. Known issues / next work

- Calibrate identity thresholds on **real multi-object** data (do not hard-code forever).
- Optional: VLM yes/no verification inside UNCERTAIN before merge (currently biases NEW + attrs).
- Clusters may need rebuild after enough observations.
- Continuous camera / taxonomy / production Docker Neo4j+Qdrant — later.
- YOLO returns 0 detections on non-COCO / low-confidence scenes — use real photos or lower conf.

---

## 9. New machine checklist

1. Clone the GitHub repo.
2. Copy `.env.example` → `.env` (never commit real secrets).
3. Create `.venv-cuda` with Python 3.13 + CUDA torch (see `scripts/bootstrap_cuda_venv.ps1` / README).
4. `pip install -r requirements.txt`
5. Install Ollama models listed above.
6. `cd frontend && npm install`
7. Start API + UI; hit `/health` → READY/DEGRADED is OK without Neo4j Docker.
8. Process `data/test_scene.jpg` twice → second pass should show **KNOWN** / reused `object_id`.
9. Ask RAG: “Where did I last see the bus?” / “What is the cell phone?”

---

## 10. Security note

Do **not** put GitHub PATs, `.env` secrets, or API keys in this repo or in chat-committed docs. If a token was pasted in chat, **revoke it** in GitHub → Settings → Developer settings → Personal access tokens after push.

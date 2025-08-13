# PubMed-Grounded GPT

FastAPI service that answers biomedical questions **only** after retrieving, chunking, ranking, and synthesizing evidence from PubMed (and PMC when available). Outputs inline PMIDs/PMCIDs with a references block.

## Endpoints
- `GET /health`
- `GET /_metrics`
- `GET /pubmed/search`
- `GET /pubmed/retrieve`
- `GET /pubmed/select`
- `POST /pubmed/answer`

## Local Dev
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\Activate)
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

import os
import requests
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load environment variables
load_dotenv()

INSTANCE = os.getenv("INSTANCE")
USERNAME = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("CLIENT_SECRET")

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- HTML Stripping Utility ---
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator="\n").strip()

# --- Paths for FAISS and lookup persistence ---
INCIDENT_INDEX_FILE = "incident_index.faiss"
INCIDENT_LOOKUP_FILE = "incident_lookup.json"
KB_INDEX_FILE = "kb_index.faiss"
KB_LOOKUP_FILE = "kb_lookup.json"

# --- Memory (In-Memory Indexes) ---
incident_index = None
incident_lookup = []
kb_index = None
kb_lookup = []

# --- Load index and lookup if available ---
def load_indexes():
    global incident_index, incident_lookup, kb_index, kb_lookup

    if os.path.exists(INCIDENT_INDEX_FILE) and os.path.exists(INCIDENT_LOOKUP_FILE):
        incident_index = faiss.read_index(INCIDENT_INDEX_FILE)
        with open(INCIDENT_LOOKUP_FILE, "r", encoding="utf-8") as f:
            incident_lookup = json.load(f)

    if os.path.exists(KB_INDEX_FILE) and os.path.exists(KB_LOOKUP_FILE):
        kb_index = faiss.read_index(KB_INDEX_FILE)
        with open(KB_LOOKUP_FILE, "r", encoding="utf-8") as f:
            kb_lookup = json.load(f)

load_indexes()

# --- Root Health Check ---
@app.get("/")
def root():
    return {"message": "ServiceNow GPT Connector is running."}

# --- Embed Closed Incidents (supports preview mode) ---
@app.post("/embed_closed_incidents")
def embed_closed_incidents(batch_size: int = 1000, max_records: int = 1000):
    global incident_index, incident_lookup

    offset = 0
    total_fetched = 0
    incident_lookup = []
    corpus = []

    while total_fetched < max_records:
        url = (
            f"{INSTANCE}/api/now/table/incident"
            f"?sysparm_query=state=7^resolved_at>=2020-01-01^ORDERBYresolved_at"
            f"&sysparm_offset={offset}"
            f"&sysparm_limit={batch_size}"
            f"&sysparm_fields=number,short_description,description,close_notes,resolved_at"
        )
        response = requests.get(url, auth=(USERNAME, PASSWORD), headers={"Accept": "application/json"})
        if response.status_code != 200:
            break

        data = response.json().get("result", [])
        if not data:
            break

        for inc in data:
            text = f"{inc.get('short_description', '')}\n{inc.get('description', '')}\n{inc.get('close_notes', '')}"
            corpus.append(text)
            incident_lookup.append({
                "number": inc.get("number"),
                "resolved_at": inc.get("resolved_at"),
                "text": text
            })

        offset += batch_size
        total_fetched += len(data)

    embeddings = model.encode(corpus, convert_to_numpy=True, batch_size=64, show_progress_bar=True)
    dim = embeddings.shape[1]
    incident_index = faiss.IndexFlatL2(dim)
    incident_index.add(embeddings)

    faiss.write_index(incident_index, INCIDENT_INDEX_FILE)
    with open(INCIDENT_LOOKUP_FILE, "w", encoding="utf-8") as f:
        json.dump(incident_lookup, f, ensure_ascii=False, indent=2)

    return {"status": "success", "indexed": len(incident_lookup)}

# --- Semantic Search on Incidents ---
@app.get("/semantic_search_incidents")
def semantic_search_incidents(q: str = Query(..., min_length=2), k: int = 5):
    global incident_index, incident_lookup

    if not incident_index or not incident_lookup:
        return JSONResponse(status_code=400, content={"error": "Call /embed_closed_incidents first."})

    query_vec = model.encode([q])
    D, I = incident_index.search(np.array(query_vec), k)
    matches = [incident_lookup[i] for i in I[0]]

    return {"query": q, "matches": matches}

# --- Embed KB Articles ---
@app.post("/embed_kb_articles")
def embed_kb_articles(batch_size: int = 1000, max_records: int = 5000):
    global kb_index, kb_lookup

    offset = 0
    total_fetched = 0
    kb_lookup = []
    corpus = []

    while total_fetched < max_records:
        url = (
            f"{INSTANCE}/api/now/table/kb_knowledge"
            f"?sysparm_query=workflow_state=published^ORDERBYpublished"
            f"&sysparm_offset={offset}"
            f"&sysparm_limit={batch_size}"
            f"&sysparm_fields=number,short_description,text,meta_description,topic,published"
        )
        response = requests.get(url, auth=(USERNAME, PASSWORD), headers={"Accept": "application/json"})
        if response.status_code != 200:
            break

        data = response.json().get("result", [])
        if not data:
            break

        for article in data:
            cleaned_text = clean_html(article.get("text", ""))
            combined = f"{article.get('short_description', '')}\n{cleaned_text}\n{article.get('meta_description', '')}"
            corpus.append(combined)
            kb_lookup.append({
                "number": article.get("number"),
                "published": article.get("published"),
                "text": combined
            })

        offset += batch_size
        total_fetched += len(data)

    embeddings = model.encode(corpus, convert_to_numpy=True, batch_size=64, show_progress_bar=True)
    dim = embeddings.shape[1]
    kb_index = faiss.IndexFlatL2(dim)
    kb_index.add(embeddings)

    faiss.write_index(kb_index, KB_INDEX_FILE)
    with open(KB_LOOKUP_FILE, "w", encoding="utf-8") as f:
        json.dump(kb_lookup, f, ensure_ascii=False, indent=2)

    return {"status": "success", "indexed": len(kb_lookup)}

# --- Semantic Search on KB Articles ---
@app.get("/semantic_search_kb")
def semantic_search_kb(q: str = Query(..., min_length=2), k: int = 5):
    global kb_index, kb_lookup

    if not kb_index or not kb_lookup:
        return JSONResponse(status_code=400, content={"error": "Call /embed_kb_articles first."})

    query_vec = model.encode([q])
    D, I = kb_index.search(np.array(query_vec), k)
    matches = [kb_lookup[i] for i in I[0]]

    return {"query": q, "matches": matches}

# --- Unified Semantic Search (KB + Incidents) ---
@app.get("/semantic_search_combined")
def semantic_search_combined(q: str = Query(..., min_length=2), k: int = 5):
    if not incident_index or not incident_lookup or not kb_index or not kb_lookup:
        return JSONResponse(status_code=400, content={"error": "Indexes not loaded. Embed incidents and KBs first."})

    query_vec = model.encode([q])
    inc_D, inc_I = incident_index.search(np.array(query_vec), k)
    kb_D, kb_I = kb_index.search(np.array(query_vec), k)

    incidents = [{"source": "incident", **incident_lookup[i], "score": float(inc_D[0][rank])} for rank, i in enumerate(inc_I[0])]
    articles = [{"source": "kb", **kb_lookup[i], "score": float(kb_D[0][rank])} for rank, i in enumerate(kb_I[0])]

    combined = sorted(incidents + articles, key=lambda x: x["score"])

    return {"query": q, "matches": combined[:k]}

# --- App Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

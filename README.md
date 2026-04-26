# StackIQ — Smarter Search for Developers

A semantic search and content-based personalized recommendation system for Stack Overflow questions.

Built as a capstone project for the Data Analytics Program at Clark University.

## What It Does

Stack Overflow has 24M+ questions but keyword search fails when different words describe the same concept — "sort array" misses "order elements in list." StackIQ solves this with a 3-stage pipeline that understands meaning, ranks by quality, and personalizes results for each developer.

## How It Works

```
User Query → Encode (MiniLM) → FAISS Retrieval (top 50) → Cross-Encoder Re-rank (top 20) → Weighted Scoring + Personalization (top 10)
```

| Stage | Component | What It Does | Speed |
|-------|-----------|-------------|-------|
| 1 | FAISS | Searches 498,644 question vectors by cosine similarity | ~3ms |
| 2 | Cross-Encoder | Re-ranks candidates by reading query + question together | ~100ms |
| 3 | Weighted Scoring | Combines relevance, votes, views, accepted answer, freshness, user preferences | ~1ms |

**Total latency: ~170ms per query**

## Results

| Approach | NDCG@5 | Avg Grade | Wins | Improvement |
|----------|--------|-----------|------|-------------|
| TF-IDF (baseline) | 0.760 | 2.84 / 5 | 1/15 | — |
| FAISS Only | 0.848 | 3.55 / 5 | 4/15 | +11.6% |
| FAISS + Cross-Encoder | 0.792 | 3.25 / 5 | 0/15 | +4.2% |
| **Full Pipeline** | **0.885** | **3.89 / 5** | **10/15** | **+16.4%** |

Evaluated across 15 test queries spanning Python, JavaScript, SQL, CSS, Java, and general programming.

## Personalization

Content-based filtering tracks user clicks and builds tag-preference profiles. A Python developer and a JavaScript developer searching "how to handle errors" get different, language-appropriate results.

## Tech Stack

- **Data**: Google BigQuery (500K questions, 1.2M answers, 30 tags)
- **Embeddings**: all-MiniLM-L6-v2 (384-dim, sentence-transformers)
- **Vector Search**: FAISS IndexFlatIP
- **Re-ranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Personalization**: SQLite + tag-preference profiles
- **Backend**: FastAPI
- **Frontend**: React (single-page app with phone-frame UI)
- **Containerization**: Docker

## Project Structure

```
├── App/
│   ├── backend.py                   # FastAPI REST API
│   ├── index.html                   # React frontend
│   ├── Dockerfile                   # Docker image definition
│   └── requirements-docker.txt      # Minimal dependencies for Docker
├── Codes/
│   ├── Recommendation_System_Data_Extraction_Code.ipynb
│   ├── Recommendation_EDA_Viz.ipynb
│   ├── embedding_generation_questions.ipynb
│   ├── faiss_index_search.ipynb
│   ├── cross_encoder_evaluation.ipynb
│   ├── improved_evaluation.ipynb
│   └── personalization.ipynb
├── Presentations/
├── Visulizations/
├── Literature_Review/
├── docker-compose.yml               # Multi-service orchestration
├── Capstone_Poster.pptx
├── Capstone_Proposal.pdf
├── requirements.txt
└── README.md
```

---

## Setup

There are two ways to run StackIQ — Docker (recommended, works on any machine) or manual setup.

---

### Option 1 — Docker (Recommended)

No Python environment setup needed. Works identically on any machine.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/sharmi1601/StackIQ.git
cd StackIQ
```

**Step 2 — Generate data files (one time only):**

Run the notebooks in this order to generate the `Dataset_Cleaned/` folder:
1. `embedding_generation_questions.ipynb` → generates embeddings
2. `faiss_index_search.ipynb` → builds FAISS index

> Note: Data extraction from BigQuery requires Google Cloud credentials. Contact the authors for a pre-built dataset.

**Step 3 — Build and run:**
```bash
docker compose build    # first time only (~10-15 minutes)
docker compose up       # starts API at localhost:8000
```

**Step 4 — Open the frontend:**

Open `App/index.html` directly in your browser.

**Daily usage (after first setup):**
```bash
docker compose up      # start
docker compose down    # stop
```

---

### Option 2 — Manual Setup

**Step 1 — Clone and install dependencies:**
```bash
git clone https://github.com/sharmi1601/StackIQ.git
cd StackIQ
conda create -n recsys python=3.10 -y
conda activate recsys
pip install -r requirements.txt
```

**Step 2 — Generate data (run notebooks in order):**
1. `Data_Extraction_Code.ipynb` → extracts data from BigQuery
2. `Recommendation_EDA_Viz.ipynb` → exploratory analysis
3. `embedding_generation_questions.ipynb` → generates embeddings
4. `faiss_index_search.ipynb` → builds FAISS index
5. `cross_encoder_evaluation.ipynb` → evaluation
6. `personalization.ipynb` → content-based personalization

**Step 3 — Run the app:**
```bash
cd App
uvicorn backend:app --reload --port 8000
```

Open `App/index.html` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/search` | Semantic search with personalization |
| GET | `/api/feed/{user_id}` | Personalized homepage feed |
| POST | `/api/signup` | Create account |
| POST | `/api/login` | Login |
| POST | `/api/interests` | Set onboarding tag preferences |
| POST | `/api/click` | Log question click for personalization |
| GET | `/api/profile/{user_id}` | User profile and history |

---

## Authors

Sharmendra Desiboyina, Tarun Kumar Jasti, Satyaki Mitra

Data Analytics Program, School of Professional Studies, Clark University

## License

MIT

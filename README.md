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

## Project Structure

```
├── App/
│   ├── backend.py              # FastAPI REST API
│   └── index.html              # React frontend
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
├── Capstone_Poster.pptx
├── Capstone_Proposal.pdf
├── requirements.txt
└── README.md
```

## Setup

**1. Clone and install dependencies:**
```bash
git clone https://github.com/sharmi1601/StackIQ.git
cd StackIQ
conda create -n recsys python=3.10 -y
conda activate recsys
pip install -r requirements.txt
```

**2. Generate data (run notebooks in order):**
- `Data_Extraction_Code.ipynb` → extracts data from BigQuery
- `Recommendation_EDA_Viz.ipynb` → exploratory analysis
- `embedding_generation_questions.ipynb` → generates embeddings
- `faiss_index_search.ipynb` → builds FAISS index
- `cross_encoder_evaluation.ipynb` → evaluation
- `personalization.ipynb` → content-based personalization

**3. Run the app:**
```bash
cd App
uvicorn backend:app --reload --port 8000
```
Open `http://localhost:8000` in your browser.

## Authors

Sharmendra Desiboyina, Tarun Kumar Jasti, Satyaki Mitra

Data Analytics Program, School of Professional Studies, Clark University

## License

MIT

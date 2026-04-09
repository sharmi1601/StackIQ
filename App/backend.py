"""
StackIQ Backend — FastAPI REST API
Wraps the complete recommendation pipeline:
  FAISS retrieval → Cross-Encoder re-ranking → Weighted scoring + Personalization

Run: uvicorn backend:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd
import faiss
import sqlite3
import os
import time
from sentence_transformers import SentenceTransformer, CrossEncoder

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "../Dataset_Cleaned"  # ⚠️ UPDATE THIS PATH
DB_PATH = os.path.join(DATA_DIR, "recsys_users.db")

# ============================================================
# LOAD PIPELINE (once at startup)
# ============================================================
print("Loading pipeline components...")

index = faiss.read_index(os.path.join(DATA_DIR, "faiss_index.bin"))
question_ids = np.load(os.path.join(DATA_DIR, "question_ids.npy"))
questions_df = pd.read_parquet(os.path.join(DATA_DIR, "questions_cleaned.parquet"))
answers_df = pd.read_parquet(os.path.join(DATA_DIR, "answers_cleaned.parquet"))
print(f"  Data: {len(questions_df):,} questions, {len(answers_df):,} answers")

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("  Models loaded")

questions_dict = questions_df.set_index("id").to_dict("index")
answers_grouped = answers_df.sort_values("answer_rank").groupby("question_id")

# Top tags for onboarding
TOP_TAGS = questions_df["primary_tag"].value_counts().head(30).index.tolist()

print("Pipeline ready!")

# ============================================================
# DATABASE SETUP
# ============================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        display_name TEXT NOT NULL,
        avatar_id INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS user_interests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        tag TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS search_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        query TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS click_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        question_id INTEGER NOT NULL,
        question_tags TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""")
    conn.commit()
    conn.close()

init_db()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_question_details(q_id):
    q = questions_dict.get(q_id)
    if q is None:
        return None
    top_answer = None
    all_answers = []
    if q_id in answers_grouped.groups:
        group = answers_grouped.get_group(q_id)
        top_answer = group.iloc[0]
        for _, ans in group.iterrows():
            all_answers.append({
                "id": int(ans["id"]),
                "body": ans["body"][:500],
                "score": int(ans["score"]),
                "is_accepted": bool(ans["is_accepted"]),
                "rank": int(ans["answer_rank"])
            })
    return {
        "question_id": q_id,
        "title": q["title"],
        "tags": q["tags"],
        "primary_tag": q["primary_tag"],
        "score": int(q["score"]),
        "views": int(q["view_count"]),
        "answer_count": int(q["answer_count"]),
        "has_accepted": bool(q["has_accepted_answer"]),
        "body": q["body"][:500],
        "creation_date": str(q["creation_date"]),
        "top_answer": top_answer["body"][:300] if top_answer is not None else "",
        "top_answer_score": int(top_answer["score"]) if top_answer is not None else 0,
        "all_answers": all_answers
    }

def build_user_tag_profile(user_id):
    conn = get_db()
    clicks = conn.execute(
        "SELECT question_tags, timestamp FROM click_history WHERE user_id = ? ORDER BY timestamp DESC",
        (user_id,)
    ).fetchall()
    
    interests = conn.execute(
        "SELECT tag FROM user_interests WHERE user_id = ?", (user_id,)
    ).fetchall()
    conn.close()
    
    tag_weights = {}
    
    # Add onboarding interests with base weight
    for row in interests:
        tag_weights[row["tag"]] = tag_weights.get(row["tag"], 0) + 0.5
    
    # Add click history with recency weighting
    total_clicks = len(clicks)
    for i, click in enumerate(clicks):
        recency = 1.0 - (i / max(total_clicks, 1)) * 0.7
        tags = [t.strip() for t in click["question_tags"].split(",") if t.strip()]
        for tag in tags:
            tag_weights[tag] = tag_weights.get(tag, 0) + recency
    
    # Normalize
    total = sum(tag_weights.values())
    if total > 0:
        tag_weights = {tag: round(w / total, 4) for tag, w in tag_weights.items()}
    
    return dict(sorted(tag_weights.items(), key=lambda x: -x[1]))

def compute_user_tag_match(question_tags, user_profile):
    if not user_profile:
        return 0.0
    tags = [t.strip() for t in question_tags.split(",") if t.strip()]
    return min(1.0, sum(user_profile.get(tag, 0) for tag in tags))

def compute_final_score(candidate, user_profile=None):
    ce_normalized = max(0, min(1, (candidate["ce_score"] + 10) / 20))
    vote_normalized = np.log1p(candidate["score"]) / np.log1p(26621)
    view_normalized = np.log1p(candidate["views"]) / np.log1p(10000000)
    accepted_bonus = 1.0 if candidate["has_accepted"] else 0.0
    ans_normalized = np.log1p(candidate.get("top_answer_score", 0)) / np.log1p(34269)
    try:
        year = pd.Timestamp(candidate["creation_date"]).year
        freshness = max(0, min(1, (year - 2008) / (2024 - 2008)))
    except:
        freshness = 0.5
    
    if user_profile:
        tag_match = compute_user_tag_match(candidate["tags"], user_profile)
        return (0.35 * ce_normalized + 0.15 * vote_normalized + 0.15 * view_normalized +
                0.10 * accepted_bonus + 0.05 * ans_normalized + 0.05 * freshness +
                0.15 * tag_match)
    else:
        return (0.40 * ce_normalized + 0.20 * vote_normalized + 0.15 * view_normalized +
                0.10 * accepted_bonus + 0.10 * ans_normalized + 0.05 * freshness)

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="StackIQ API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================
class SignupRequest(BaseModel):
    username: str
    display_name: str
    avatar_id: Optional[int] = 1

class LoginRequest(BaseModel):
    username: str

class InterestsRequest(BaseModel):
    user_id: int
    tags: List[str]

class SearchRequest(BaseModel):
    query: str
    user_id: Optional[int] = None
    limit: Optional[int] = 5

class ClickRequest(BaseModel):
    user_id: int
    question_id: int
    question_tags: str

# ============================================================
# AUTH ENDPOINTS
# ============================================================
@app.post("/api/signup")
def signup(req: SignupRequest):
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("INSERT INTO users (username, display_name, avatar_id) VALUES (?, ?, ?)",
                  (req.username, req.display_name, req.avatar_id))
        conn.commit()
        user_id = c.lastrowid
        return {"user_id": user_id, "username": req.username, "display_name": req.display_name, "avatar_id": req.avatar_id}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        conn.close()

@app.post("/api/login")
def login(req: LoginRequest):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (req.username,)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    has_interests = False
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) as c FROM user_interests WHERE user_id = ?", (user["id"],)).fetchone()
    conn.close()
    has_interests = count["c"] > 0
    
    return {
        "user_id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "avatar_id": user["avatar_id"],
        "has_interests": has_interests
    }

# ============================================================
# ONBOARDING
# ============================================================
@app.get("/api/tags")
def get_tags():
    return {"tags": TOP_TAGS}

@app.post("/api/interests")
def set_interests(req: InterestsRequest):
    conn = get_db()
    # Clear existing interests
    conn.execute("DELETE FROM user_interests WHERE user_id = ?", (req.user_id,))
    for tag in req.tags:
        conn.execute("INSERT INTO user_interests (user_id, tag) VALUES (?, ?)", (req.user_id, tag))
    conn.commit()
    conn.close()
    return {"status": "ok", "tags_saved": len(req.tags)}

# ============================================================
# SEARCH (MAIN PIPELINE)
# ============================================================
@app.post("/api/search")
def search(req: SearchRequest):
    start = time.time()
    
    # Get user profile
    user_profile = None
    if req.user_id:
        user_profile = build_user_tag_profile(req.user_id)
    
    # Stage 1: FAISS
    query_vec = bi_encoder.encode([req.query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    faiss_scores, indices = index.search(query_vec, 50)
    
    candidates = []
    for fs, idx in zip(faiss_scores[0], indices[0]):
        q_id = int(question_ids[idx])
        details = get_question_details(q_id)
        if details:
            details["faiss_score"] = float(fs)
            candidates.append(details)
    
    # Stage 2: Cross-Encoder
    pairs = [[req.query, c["title"]] for c in candidates]
    ce_scores = cross_encoder.predict(pairs)
    for i, c in enumerate(candidates):
        c["ce_score"] = float(ce_scores[i])
    
    candidates.sort(key=lambda x: x["ce_score"], reverse=True)
    candidates = candidates[:20]
    
    # Stage 3: Weighted scoring
    for c in candidates:
        c["final_score"] = compute_final_score(c, user_profile)
        c["tag_match"] = compute_user_tag_match(c["tags"], user_profile) if user_profile else 0.0
    
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    results = candidates[:req.limit]
    
    # Log search
    if req.user_id:
        conn = get_db()
        conn.execute("INSERT INTO search_history (user_id, query) VALUES (?, ?)", (req.user_id, req.query))
        conn.commit()
        conn.close()
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "query": req.query,
        "results": results,
        "latency_ms": round(elapsed, 1),
        "personalized": user_profile is not None and len(user_profile) > 0
    }

# ============================================================
# FEED (HOMEPAGE)
# ============================================================
@app.get("/api/feed/{user_id}")
def get_feed(user_id: int, limit: int = 10, refresh: int = 0):
    """refresh parameter changes the random seed to get different questions each time."""
    import random
    rng = random.Random(user_id * 1000 + refresh)
    
    user_profile = build_user_tag_profile(user_id)
    
    if user_profile:
        # PART 1: Personalized questions (70% of feed)
        personalized_count = max(7, limit - 3)
        top_tags = list(user_profile.keys())[:10]
        mask = questions_df["primary_tag"].isin(top_tags)
        user_questions = questions_df[mask].copy()
        user_questions["feed_score"] = (
            np.log1p(user_questions["score"]) * 0.4 +
            np.log1p(user_questions["view_count"]) * 0.3 +
            user_questions["has_accepted_answer"].astype(float) * 0.3
        )
        user_questions["tag_boost"] = user_questions["primary_tag"].map(lambda t: user_profile.get(t, 0))
        user_questions["feed_score"] = user_questions["feed_score"] * (1 + user_questions["tag_boost"])
        
        pool_size = limit * 10
        top_pool = user_questions.nlargest(pool_size, "feed_score")
        pool_rows = list(top_pool.iterrows())
        rng.shuffle(pool_rows)
        
        results = []
        seen_ids = set()
        for _, row in pool_rows:
            if len(results) >= personalized_count:
                break
            q_id = int(row["id"])
            if q_id in seen_ids:
                continue
            details = get_question_details(q_id)
            if details:
                details["feed_reason"] = f"Because you like {row['primary_tag']}"
                results.append(details)
                seen_ids.add(q_id)
        
        # PART 2: Discovery questions from OTHER tags (30% of feed)
        discovery_count = limit - len(results)
        if discovery_count > 0:
            other_tags = [t for t in TOP_TAGS if t not in top_tags]
            rng.shuffle(other_tags)
            for tag in other_tags[:discovery_count + 2]:
                if len(results) >= limit:
                    break
                tag_qs = questions_df[questions_df["primary_tag"] == tag]
                top_tag_qs = tag_qs.nlargest(10, "score")
                sample_rows = list(top_tag_qs.iterrows())
                rng.shuffle(sample_rows)
                for _, row in sample_rows[:1]:
                    q_id = int(row["id"])
                    if q_id not in seen_ids:
                        details = get_question_details(q_id)
                        if details:
                            details["feed_reason"] = f"Discover {tag}"
                            results.append(details)
                            seen_ids.add(q_id)
        
        return {"feed_type": "personalized", "results": results[:limit], "profile": user_profile}
    else:
        results = []
        top_tags = questions_df["primary_tag"].value_counts().head(5).index.tolist()
        for tag in top_tags:
            tag_qs = questions_df[questions_df["primary_tag"] == tag].nlargest(6, "score")
            pool = list(tag_qs.iterrows())
            rng.shuffle(pool)
            for _, row in pool[:2]:
                details = get_question_details(int(row["id"]))
                if details:
                    details["feed_reason"] = f"Popular in {tag}"
                    results.append(details)
        rng.shuffle(results)
        return {"feed_type": "cold_start", "results": results[:limit], "profile": {}}

# ============================================================
# QUESTION DETAIL
# ============================================================
@app.get("/api/question/{question_id}")
def get_question(question_id: int):
    details = get_question_details(question_id)
    if not details:
        raise HTTPException(status_code=404, detail="Question not found")
    return details

# ============================================================
# CLICK TRACKING
# ============================================================
@app.post("/api/click")
def log_click(req: ClickRequest):
    conn = get_db()
    conn.execute(
        "INSERT INTO click_history (user_id, question_id, question_tags) VALUES (?, ?, ?)",
        (req.user_id, req.question_id, req.question_tags)
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}

# ============================================================
# USER PROFILE & HISTORY
# ============================================================
@app.get("/api/profile/{user_id}")
def get_profile(user_id: int):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    searches = conn.execute(
        "SELECT query, timestamp FROM search_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20",
        (user_id,)
    ).fetchall()
    
    clicks = conn.execute(
        "SELECT question_id, question_tags, timestamp FROM click_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20",
        (user_id,)
    ).fetchall()
    conn.close()
    
    profile = build_user_tag_profile(user_id)
    
    # Get question titles for click history
    click_details = []
    for click in clicks:
        q = questions_dict.get(click["question_id"])
        click_details.append({
            "question_id": click["question_id"],
            "title": q["title"] if q else "Unknown question",
            "tags": click["question_tags"],
            "timestamp": click["timestamp"]
        })
    
    return {
        "user_id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "avatar_id": user["avatar_id"],
        "tag_profile": profile,
        "search_history": [{"query": s["query"], "timestamp": s["timestamp"]} for s in searches],
        "click_history": click_details
    }

# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "questions": len(questions_df),
        "answers": len(answers_df),
        "index_vectors": index.ntotal,
        "tags": len(TOP_TAGS)
    }


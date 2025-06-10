import os
import json
import numpy as np
import openai
import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ─── Environment ───────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMB_MODEL      = os.getenv("EMBEDDING_MODEL")
LLM_MODEL      = "gpt-3.5-turbo"
openai.api_key = OPENAI_API_KEY

# ─── Load FAISS index + metadata ────────────────────────────────────────────
index = faiss.read_index("embeddings/index.faiss")
with open("embeddings/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)  # each entry: {"text": "..."}

# ─── FastAPI application ───────────────────────────────────────────────────
app = FastAPI()

# Enable CORS for your Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
      "https://www.viswabonam.com",
      "https://viswabonam.com"
    ],
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    top_k: int = 5

# ─── Helper: Embed text ─────────────────────────────────────────────────────
def embed_text(text: str) -> np.ndarray:
    resp = openai.embeddings.create(input=text, model=EMB_MODEL)
    vec = resp.data[0].embedding
    return np.array(vec, dtype="float32")

# ─── /chat endpoint ─────────────────────────────────────────────────────────
@app.post("/chat")
def chat(query: Query):
    # 1) Embed user question
    q_vec = embed_text(query.question)

    # 2) FAISS search (L2 distances)
    distances, indices = index.search(q_vec.reshape(1, -1), query.top_k)
    best_dist = float(distances[0][0])

    # Grounding threshold: tune as needed
    DIST_THRESHOLD = 50.0
    if best_dist > DIST_THRESHOLD:
        return {
            "answer": "I’m sorry, I don’t have enough information to answer that.",
            "sources": []
        }

    # 3) Retrieve the top-k context texts
    contexts = [metadata[i]["text"] for i in indices[0]]
    context_str = "\n---\n".join(contexts)

    # 4) Build prompt (avoid backslashes inside f-string)
    prompt = (
        "You are an assistant that answers based on the provided contexts.\n"
        "Answer concisely and do NOT repeat the full context—only quote brief snippets if needed.\n\n"
        "Context:\n" + context_str + "\n\n"
        f"Question: {query.question}\n"
        "Answer:\n"
    )

    # 5) Call the Chat model with first-person instructions
    chat_resp = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the person whose resume and FAQs these are. "
                    "When you answer, speak in the first person (\"I ...\"). "
                    "Answer questions as if you yourself authored the resume."
                )
            },
            {"role": "user", "content": prompt}
        ]
    )
    answer = chat_resp.choices[0].message.content.strip()

    # 6) Return the answer and source IDs
    return {
        "answer": answer,
        "sources": [f"chunk_{i}" for i in indices[0]]
    }

# ─── Run with Uvicorn ───────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

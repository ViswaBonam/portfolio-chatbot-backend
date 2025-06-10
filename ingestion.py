import os
import json
from dotenv import load_dotenv
import openai
import faiss
import numpy as np
from pypdf import PdfReader

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EMB_MODEL        = os.getenv('EMBEDDING_MODEL')
openai.api_key   = OPENAI_API_KEY

# Text chunking helper
def chunk_text(text, max_chars=1000):
    paras, chunks, curr = text.split("\n\n"), [], ""
    for p in paras:
        if len(curr) + len(p) < max_chars:
            curr += p + "\n\n"
        else:
            chunks.append(curr)
            curr = p + "\n\n"
    if curr:
        chunks.append(curr)
    return chunks

# Load resume PDF into chunks
def load_resume(path):
    reader = PdfReader(path)
    full = "".join(page.extract_text() or "" for page in reader.pages)
    return chunk_text(full)

# Load FAQs text into chunks
def load_faqs(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return chunk_text(text)

# Build FAISS index from document chunks
def build_index(docs, index_path='embeddings/index.faiss', meta_path='embeddings/metadata.json'):
    dims = 1536
    index = faiss.IndexFlatL2(dims)
    metadata, embeddings = [], []

    for doc in docs:
        resp = openai.embeddings.create(input=doc, model=EMB_MODEL)
        em = resp.data[0].embedding
        embeddings.append(em)
        metadata.append({'text': doc})

    xb = np.array(embeddings, dtype='float32')
    index.add(xb)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)

if __name__ == '__main__':
    resume_chunks = load_resume('data/Viswa_Bonam_Resume.pdf')
    faq_chunks    = load_faqs('data/faqs.txt')
    build_index(resume_chunks + faq_chunks)
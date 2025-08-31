import fitz  # PyMuPDF
from typing import List
import faiss
import numpy as np
import os
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import sqlite3

# model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = model.encode(list_of_chunks)  # embeds each text chunk into a 384-d vecto


pdf_paths = [
    "data/2508.15711v1.pdf",
    "data/2508.15721v1.pdf",
    "data/2508.15746v1.pdf",
    "data/2508.15754v1.pdf",
    "data/2508.15760v1.pdf",
]

MODEL_NAME = "all-MiniLM-L6-v2"
SQLITE_DB = "mydata.db"


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all text as a single string.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        page_text = page.get_text()
        pages.append(page_text)
    full_text = "\n".join(pages)
    return full_text


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i : i + max_tokens]
        chunks.append(" ".join(chunk))
    return chunks


# def build_sqlite_and_fts(list_of_chunks, corpus_texts, pdf_paths):
def build_sqlite_and_fts(pdf_paths):
    if os.path.exists(SQLITE_DB):
        os.remove(SQLITE_DB)
    # conn = sqlite3.connect(SQLITE_DB)
    conn = sqlite3.connect(SQLITE_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # CREATE TABLE chunks (
    #     id        INTEGER PRIMARY KEY,
    #     doc_id    INTEGER NOT NULL,
    #     chunk_idx INTEGER NOT NULL,
    #     content   TEXT NOT NULL,
    #     FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    # );

    conn.executescript(
        """
    CREATE TABLE documents (
        doc_id   INTEGER PRIMARY KEY,
        title    TEXT,
        author   TEXT,
        year     INTEGER,
        keywords TEXT,
        path     TEXT
    );
    CREATE TABLE chunks (
        id        INTEGER PRIMARY KEY,
        doc_id    INTEGER NOT NULL,
        chunk_idx INTEGER NOT NULL,
        content   TEXT NOT NULL,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    );
    CREATE VIRTUAL TABLE doc_chunks USING fts5(
        content,
        content='chunks',
        content_rowid='id'
    );
    """
    )

    conn.executemany(
        "INSERT INTO documents(doc_id, title, path) VALUES (?, ?, ?)",
        [(i + 1, os.path.basename(p), p) for i, p in enumerate(pdf_paths)],
    )

    list_of_chunks: List[str] = []
    chunk_rows = []
    next_id = 1
    for doc_id, pdf in enumerate(pdf_paths, start=1):
        text = extract_text_from_pdf(pdf)
        parts = chunk_text(text, max_tokens=512, overlap=50)
        for cidx, content in enumerate(parts):
            list_of_chunks.append(content)
            # id = next_id ensures FAISS row i ↔ chunks.id = i+1 later, if you embed in same order
            chunk_rows.append((next_id, doc_id, cidx, content))
            next_id += 1

    conn.executemany(
        "INSERT INTO chunks(id, doc_id, chunk_idx, content) VALUES (?,?,?,?)",
        chunk_rows,
    )

    conn.execute(
        "INSERT INTO doc_chunks(rowid, content) SELECT id, content FROM chunks;"
    )

    # conn.executemany(
    #     "INSERT INTO doc_chunks(content, doc_id, chunk_idx) VALUES (?, ?, ?)",
    #     [
    #         (text, doc_id, cidx)
    #         for (doc_id, cidx), text in zip(chunk_doc_pairs, list_of_chunks)
    #     ],
    # )

    # for doc_id, doc_text in enumerate(corpus_texts, start=1):
    #     parts = chunk_text(doc_text)
    #     for cidx, content in enumerate(parts):
    #         # list_of_chunks.append(content)
    #         conn.execute(
    #             "INSERT INTO doc_chunks(content, doc_id, chunk_idx) VALUES (?, ?, ?)",
    #             (content, doc_id, cidx),
    #         )

    # list_of_chunks: List[str] = []
    # chunk_rows = []
    # next_id = 1
    # for doc_id, pdf in enumerate(pdf_paths, start=1):
    # text = extract_text_from_pdf(pdf)
    # parts = chunk_text(text, max_tokens=512, overlap=50)
    # for cidx, content in enumerate(parts):
    #     list_of_chunks.append(content)
    #     # id = next_id ensures FAISS row i ↔ chunks.id = i+1 later, if you embed in same order
    #     chunk_rows.append((next_id, doc_id, cidx, content))
    #     next_id += 1

    # conn.executemany(
    #     "INSERT INTO chunks(id, doc_id, chunk_idx, content) VALUES (?,?,?,?)",
    #     list_of_chunks,
    # )

    conn.commit()
    return conn, list_of_chunks


def fetch_chunk_payload(conn, chunk_id: int):
    row = conn.execute(
        "SELECT c.id AS chunk_id, c.doc_id, c.chunk_idx, c.content, d.title, d.path "
        "FROM chunks c JOIN documents d ON c.doc_id=d.doc_id WHERE c.id=?",
        (chunk_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "chunk_id": int(row["chunk_id"]),
        "doc_id": int(row["doc_id"]),
        "chunk_idx": int(row["chunk_idx"]),
        "title": row["title"],
        "path": row["path"],
        "chunk": row["content"],
    }


def keyword_topk(conn, q: str, k: int):
    try:
        rows = conn.execute(
            "SELECT c.id, bm25(f) AS score "
            "FROM doc_chunks f JOIN chunks c ON f.rowid=c.id "
            "WHERE f MATCH ? ORDER BY score LIMIT ?;",
            (q, k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for r in rows:
        payload = fetch_chunk_payload(conn, int(r["id"]))
        if payload:
            payload["score"] = float(r["score"])
            out.append(payload)
    return out


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


model = SentenceTransformer(MODEL_NAME)

# corpus_texts = []
# for p in pdf_paths:
#     if not os.path.exists(p):
#         print(f"Missing PDF: {p}")
#     else:
#         corpus_texts.append(extract_text_from_pdf(p))

# # full_corpus = "\n\n".join(corpus_texts)


# list_of_chunks_outer = []
# for doc in corpus_texts:
#     list_of_chunks_outer.extend(chunk_text(doc))


conn, list_of_chunks = build_sqlite_and_fts(pdf_paths)
# print("SQLite ready with", len(list_of_chunks), "chunks.")

embeddings = model.encode(list_of_chunks, convert_to_numpy=True).astype(
    np.float32
)  # embeds each text chunk into a 384-d vecto


# Assume embeddings is a 2D numpy array of shape (num_chunks, dim)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # using a simple L2 index
index.add(np.array(embeddings))  # add all chunk vectors

# # Example: search for a query embedding
# query = "What is retrieval augmented generation?"
# query_embedding = model.encode([query])  # get embedding for the query (shape: [1, dim])
# k = 3
# distances, indices = index.search(query_embedding, k)
# # indices[0] holds the top-k chunk indices

# for i in indices[0]:
#     print(list_of_chunks[i])


app = FastAPI()


# async
@app.get("/search")
def search(q: str):
    """
    Receive a query 'q', embed it, retrieve top-3 passages, and return them.
    """
    # TODO: Embed the query 'q' using your embedding model
    query_vector = model.encode([q])[0]  # e.g., model.encode([q])[0]
    query_vector = np.asarray(query_vector, dtype=np.float32)
    # Perform FAISS search
    k = 3
    distances, indices = index.search(np.array([query_vector]), k)
    # Retrieve the corresponding chunks (assuming 'chunks' list and 'indices' shape [1, k])
    results = []
    for idx in indices[0]:
        results.append(list_of_chunks[idx])
    return {"query": q, "results": results}


@app.get("/keyword_search")
def keyword_search(q: str, k: int = 3):
    if not q:
        print("keyword_search error: q is required")
        # raise HTTPException(400, "q is required")
    k = max(1, min(k, index.ntotal))
    results = keyword_topk(conn, q, k)
    return {"mode": "keyword", "query": q, "k": k, "results": results}


@app.get("/hybrid_search")
def hybrid_search(q: str, k: int = 3, k_vec: int = 10, k_key: int = 10):
    if not q:
        print("hybrid_search error: q is required")
        # raise HTTPException(400, "q is required")
    k = max(1, min(k, index.ntotal))
    k_vec = max(k, k_vec)
    k_key = max(k, k_key)

    qv = model.encode([q], convert_to_numpy=True).astype(np.float32)
    D, I = index.search(qv, k_vec)
    vec_ids = [int(i) + 1 for i in I[0]]

    key_payloads = keyword_topk(conn, q, k_key)
    key_ids = [p["chunk_id"] for p in key_payloads]

    r_vec = {cid: r + 1 for r, cid in enumerate(vec_ids)}
    r_key = {cid: r + 1 for r, cid in enumerate(key_ids)}

    all_ids = set(vec_ids) | set(key_ids)
    scored = []
    for cid in all_ids:
        s = 0.0
        if cid in r_vec:
            s += rrf_score(r_vec[cid])
        if cid in r_key:
            s += rrf_score(r_key[cid])
        scored.append((cid, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    out = []
    for cid, _ in scored[:k]:
        payload = fetch_chunk_payload(conn, cid)
        if payload:
            out.append(payload)
    return {"mode": "hybrid_rrf", "query": q, "k": k, "results": out}

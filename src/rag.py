"""
RAG (retrieval-augmented) technical Q&A over the platform documentation
corpus, using FAISS for the vector index.

Generation backend is pluggable:
  - If GEMINI_API_KEY is set in the environment, real calls are made to the
    Gemini API (google-generativeai), with the retrieved doc chunks injected
    into the prompt as grounding context.
  - Otherwise, an extractive fallback constructs the answer directly from the
    top-ranked retrieved chunk(s) -- no network call, fully deterministic,
    and always grounded (it literally can't hallucinate since it only
    returns retrieved text). This is the path exercised by the test suite
    and the dashboard in this build environment, since outbound calls to
    generativelanguage.googleapis.com are not reachable from this sandbox.

Either way, every answer returns its source doc_ids, so a human agent can
verify the answer against the underlying documentation before it reaches a
customer -- this is deliberate: the JD asks for AI that speeds up
resolution, not AI that answers customers unsupervised.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import faiss

from .embeddings import Embedder, get_embedder

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    section: str
    text: str
    score: float


@dataclass
class RAGAnswer:
    query: str
    answer: str
    sources: list[RetrievedDoc]
    generation_backend: str


def load_docs(path: Path = None) -> list[dict]:
    path = path or (DATA_DIR / "platform_docs.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class RAGIndex:
    def __init__(self, embedder: Embedder = None):
        self.embedder = embedder or get_embedder("tfidf-svd")
        self.docs: list[dict] = []
        self.index: faiss.Index | None = None

    def build(self, docs: list[dict] = None) -> "RAGIndex":
        self.docs = docs or load_docs()
        corpus = [f"{d['title']}. {d['text']}" for d in self.docs]
        self.embedder.fit(corpus)
        vecs = self.embedder.embed(corpus)
        self.index = faiss.IndexFlatIP(vecs.shape[1])  # cosine sim via normalized IP
        self.index.add(vecs)
        return self

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedDoc]:
        if self.index is None:
            raise RuntimeError("Call build() before retrieve().")
        qvec = self.embedder.embed([query])
        scores, idxs = self.index.search(qvec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            d = self.docs[idx]
            results.append(RetrievedDoc(
                doc_id=d["doc_id"], title=d["title"], section=d["section"],
                text=d["text"], score=float(score),
            ))
        return results


def _extractive_answer(query: str, retrieved: list[RetrievedDoc]) -> str:
    if not retrieved or retrieved[0].score < 0.05:
        return (
            "I couldn't find anything in the platform documentation that "
            "directly answers this -- routing to a human agent rather than "
            "guessing."
        )
    top = retrieved[0]
    extra = ""
    if len(retrieved) > 1 and retrieved[1].score > 0.3:
        extra = f" You may also want to check '{retrieved[1].title}' for related detail."
    return f"{top.text}{extra}"


def _gemini_answer(query: str, retrieved: list[RetrievedDoc]) -> str:
    """Real Gemini call. Not exercised in this sandbox (no outbound access
    to generativelanguage.googleapis.com + no API key configured here) --
    included so the wiring is real and swaps in with just an env var
    elsewhere.
    """
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    context = "\n\n".join(f"[{d.doc_id}] {d.title}: {d.text}" for d in retrieved)
    prompt = (
        "You are a support assistant for a trading platform. Answer the "
        "user's question using ONLY the context below. If the context "
        "doesn't cover it, say so rather than guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


def answer_question(index: RAGIndex, query: str, k: int = 3) -> RAGAnswer:
    retrieved = index.retrieve(query, k=k)

    if os.environ.get("GEMINI_API_KEY"):
        try:
            answer = _gemini_answer(query, retrieved)
            backend = "gemini-1.5-flash"
        except Exception as e:  # pragma: no cover - network path, not exercised here
            answer = _extractive_answer(query, retrieved)
            backend = f"extractive_fallback (gemini call failed: {e})"
    else:
        answer = _extractive_answer(query, retrieved)
        backend = "extractive_fallback"

    return RAGAnswer(query=query, answer=answer, sources=retrieved, generation_backend=backend)


def build_default_index() -> RAGIndex:
    return RAGIndex().build()


if __name__ == "__main__":
    idx = build_default_index()
    test_queries = [
        "why did my order get rejected",
        "how does margin call liquidation work",
        "what's the API rate limit for placing orders",
        "how long do withdrawals take",
        "can I use MetaTrader 5",
    ]
    for q in test_queries:
        result = answer_question(idx, q)
        print(f"\nQ: {q}")
        print(f"A ({result.generation_backend}): {result.answer[:200]}...")
        print("Sources:", [f"{s.doc_id} ({s.score:.3f})" for s in result.sources])

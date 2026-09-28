"""
Pluggable text-embedding backends for the RAG retrieval layer.

Design note (documented honestly, not hidden): this project ships two
embedding backends behind one interface:

  - TfidfSvdEmbedder        : fully local/offline, no network or model
                               download required. Fit once on the doc corpus,
                               deterministic, fast. This is the DEFAULT and
                               the one actually exercised by the test suite
                               and the dashboard in this environment.
  - SentenceTransformerEmbedder : wraps `sentence-transformers` (real,
                               tested-for-import code) for a proper dense
                               transformer embedding. It needs to download a
                               model from Hugging Face on first use, which
                               requires outbound network access. In the
                               sandbox this was built in, that download is
                               blocked by the network allowlist (HTTP 403),
                               so this backend is wired and unit-testable for
                               its interface, but NOT exercised end-to-end
                               here. Swapping EMBEDDING_BACKEND="sentence-
                               transformer" in config.py enables it wherever
                               network access to huggingface.co is available.

Both implement the same `Embedder` interface so `rag.py` doesn't care which
one is active.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class Embedder:
    def fit(self, texts: list[str]) -> "Embedder":
        raise NotImplementedError

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array, L2-normalized per row."""
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class TfidfSvdEmbedder(Embedder):
    """TF-IDF -> truncated SVD (latent semantic) embedding. Fully offline."""

    def __init__(self, n_components: int = 128, random_state: int = 42):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), stop_words="english")
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfSvdEmbedder":
        # SVD component count can't exceed min(n_samples, n_features) - 1
        n_components = min(self.n_components, max(2, len(texts) - 1))
        if n_components != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before embed().")
        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf).astype("float32")
        return _l2_normalize(vecs)

    @property
    def dim(self) -> int:
        return self.svd.n_components


class SentenceTransformerEmbedder(Embedder):
    """Real sentence-transformers backend. Requires network to fetch the
    model on first use (blocked in this build sandbox -- see module
    docstring). Interface-tested but not exercised end-to-end here.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        # No fitting needed -- the transformer is pretrained. We still load
        # it here (rather than lazily) so callers can fail fast if the model
        # isn't reachable.
        self._load()
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(texts, convert_to_numpy=True).astype("float32")
        return _l2_normalize(vecs)

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()


def get_embedder(backend: str = "tfidf-svd") -> Embedder:
    if backend == "tfidf-svd":
        return TfidfSvdEmbedder()
    if backend == "sentence-transformer":
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")

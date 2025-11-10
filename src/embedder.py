# src/embedder.py
import os
import numpy as np
try:
    import openai
    OPENAI_INSTALLED = True
except Exception:
    OPENAI_INSTALLED = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

EMBED_DIM = 1536  # target dim for text-embedding-3-large

def openai_embed_texts(texts, model="text-embedding-3-large", batch_size=64):
    if not OPENAI_INSTALLED:
        raise RuntimeError("openai package not installed in environment.")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set as environment variable.")
    openai.api_key = key
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = openai.Embeddings.create(model=model, input=batch)
        embs = [item["embedding"] for item in resp["data"]]
        all_embs.extend(embs)
    return np.array(all_embs)

def fallback_embed_texts_tfidf_svd(texts, dim=EMBED_DIM, random_state=42, max_features=4000):
    tf = TfidfVectorizer(ngram_range=(1,2), max_features=max_features)
    X = tf.fit_transform(texts)
    n_comp = min(dim, max(1, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    Xr = svd.fit_transform(X)
    norms = np.linalg.norm(Xr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xr = Xr / norms
    if Xr.shape[1] < dim:
        pad = np.zeros((Xr.shape[0], dim - Xr.shape[1]))
        Xr = np.hstack([Xr, pad])
    return Xr

def embed_texts_auto(texts, prefer_openai=False):
    if prefer_openai:
        try:
            return openai_embed_texts(texts)
        except Exception as e:
            print("OpenAI embedding failed, falling back. Reason:", e)
    return fallback_embed_texts_tfidf_svd(texts)

# tests/test_embedding_and_search.py
from src.data_prep import load_and_prep
from src.embedder import fallback_embed_texts_tfidf_svd, embed_texts_auto
from src.searcher import search_top_k

def test_fallback_embedding_shape():
    df = load_and_prep()
    texts = df['embed_text'].tolist()
    embs = fallback_embed_texts_tfidf_svd(texts, dim=128)
    assert embs.shape[0] == len(texts)

def test_search_top_k():
    df = load_and_prep()
    texts = df['embed_text'].tolist()
    embs = fallback_embed_texts_tfidf_svd(texts, dim=128)
    q_emb = embs[0]
    idxs, scores, best = search_top_k(q_emb, embs, k=3)
    assert len(idxs) == 3

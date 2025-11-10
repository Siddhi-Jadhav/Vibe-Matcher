# src/searcher.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from time import perf_counter

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

def search_top_k(q_emb, emb_matrix, k=3):
    q_emb = q_emb.reshape(1, -1)
    sims = cosine_similarity(q_emb, emb_matrix)[0]
    idxs = np.argsort(sims)[::-1][:k]
    return idxs, sims[idxs], float(np.max(sims))

def match_query_workflow(query, df_products, emb_matrix, embed_func, top_k=3, threshold=0.7):
    out = { "query": query }
    t0 = perf_counter()
    q_emb = embed_func([query])[0]
    t_embed = perf_counter() - t0
    t1 = perf_counter()
    idxs, scores, best = search_top_k(q_emb, emb_matrix, k=top_k)
    t_sim = perf_counter() - t1
    results = []
    for idx, score in zip(idxs, scores):
        row = df_products.iloc[idx].to_dict()
        results.append({ "id": int(row.get("id")), "name": row.get("name"), "desc": row.get("desc"), "tags": row.get("tags"), "score": float(score) })
    fallback = None
    if best < threshold:
        fallback = f"No strong match (best score {best:.3f} < threshold {threshold}). Consider broadening query or showing curated picks."
    out.update({
        "results": results,
        "best_score": float(best),
        "fallback": fallback,
        "timings": { "embedding_s": t_embed, "similarity_s": t_sim, "total_s": (t_embed + t_sim) }
    })
    return out

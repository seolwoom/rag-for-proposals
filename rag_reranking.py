"""Lazy local cross-encoder reranking of children before parent expansion."""
from functools import lru_cache
import math

RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'
CANDIDATE_K = 20
PARENT_K = 4
MAX_LENGTH = 1024


@lru_cache(maxsize=1)
def load_reranker():
    # Download once on first retrieval; CUDA is used when available, otherwise CPU.
    from sentence_transformers import CrossEncoder
    from torch.nn import Identity
    return CrossEncoder(RERANKER_MODEL, max_length=MAX_LENGTH,
                        activation_fn=Identity(), trust_remote_code=False)


def rerank_children(query, children, model=None, candidate_k=CANDIDATE_K):
    """Score candidates, using overlapping token windows for oversized tables/text.

    A child's score is its maximum window score. Original content and metadata
    are preserved. Scores are ranking logits, not calibrated probabilities.
    """
    from langchain_core.documents import Document
    if candidate_k < 1:
        raise ValueError('candidate_k must be positive')
    candidates, seen = [], set()
    for child in children:
        key = (child.metadata['source'], child.metadata['parent_id'], child.metadata['chunk_id'])
        if key not in seen:
            candidates.append(child)
            seen.add(key)
        if len(candidates) == candidate_k:
            break
    if not candidates:
        return []
    model = model if model is not None else load_reranker()
    tokenizer = model.tokenizer
    query_tokens = tokenizer.encode(query, add_special_tokens=False)
    # Reserve passage space even for very long queries.
    scoring_query = query if len(query_tokens) <= 256 else tokenizer.decode(query_tokens[:256])
    query_length = len(tokenizer.encode(scoring_query, add_special_tokens=False))
    budget = MAX_LENGTH - query_length - tokenizer.num_special_tokens_to_add(pair=True) - 8
    if budget < 1:
        raise ValueError('No passage token budget remains')
    pairs, owners, counts = [], [], []
    for index, child in enumerate(candidates):
        tokens = tokenizer.encode(child.page_content, add_special_tokens=False)
        windows = [child.page_content] if len(tokens) <= budget else [
            tokenizer.decode(tokens[start:start + budget])
            for start in range(0, len(tokens), max(1, budget - min(128, budget // 4)))
        ]
        counts.append(len(windows))
        pairs.extend((scoring_query, text) for text in windows)
        owners.extend([index] * len(windows))
    predictions = model.predict(pairs, batch_size=8, show_progress_bar=False)
    if len(predictions) != len(pairs):
        raise ValueError('Reranker returned an unexpected number of scores')
    scores = [-math.inf] * len(candidates)
    best_passages = [''] * len(candidates)
    for owner, value, pair in zip(owners, predictions, pairs):
        score = float(value)
        if not math.isfinite(score):
            raise ValueError('Reranker returned a non-finite score')
        if score > scores[owner]:
            scores[owner] = score
            best_passages[owner] = pair[1]
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [Document(page_content=candidates[i].page_content, metadata={
        **candidates[i].metadata, 'reranker_score': scores[i],
        'reranker_model': RERANKER_MODEL, 'reranker_window_count': counts[i],
        'reranker_best_passage': best_passages[i],
    }) for i in order]


def rerank_to_parents(query, children, parents, model=None,
                      candidate_k=CANDIDATE_K, parent_k=PARENT_K):
    """Rank parents by their best matching child; attach per-query evidence."""
    from langchain_core.documents import Document
    if parent_k < 1:
        raise ValueError('parent_k must be positive')
    ranked = rerank_children(query, children, model=model, candidate_k=candidate_k)
    results, seen = [], set()
    for child in ranked:
        parent_id = child.metadata['parent_id']
        if parent_id in seen:
            continue
        parent = parents[parent_id]
        child_start = child.metadata.get('start_index')
        if child_start is None:
            child_start = max(0, parent.page_content.find(child.page_content))
        passage_offset = child.page_content.find(child.metadata['reranker_best_passage'])
        results.append(Document(page_content=parent.page_content, metadata={
            **parent.metadata, 'reranker_score': child.metadata['reranker_score'],
            'reranker_model': RERANKER_MODEL,
            'matched_child_id': child.metadata['chunk_id'],
            'matched_child_pages': child.metadata.get('pages', []),
            'matched_start_index': child_start + max(0, passage_offset),
            'matched_end_index': child.metadata.get('end_index', child_start + len(child.page_content)),
        }))
        seen.add(parent_id)
        if len(results) == parent_k:
            break
    return results

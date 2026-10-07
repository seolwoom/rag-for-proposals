"""Bounded context around the child that caused a parent to be retrieved."""
from functools import lru_cache

CONTEXT_TOKEN_BUDGET = 8000
PER_PARENT_TOKEN_BUDGET = 2000
OUTPUT_TOKEN_LIMIT = 4096
LLM_MODEL = 'gpt-5-mini'


@lru_cache(maxsize=1)
def encoding():
    import tiktoken
    return tiktoken.get_encoding('o200k_base')


def count_tokens(text):
    return len(encoding().encode(text, disallowed_special=()))


def format_context(docs, token_budget=CONTEXT_TOKEN_BUDGET,
                   per_parent_budget=PER_PARENT_TOKEN_BUDGET, debug=False):
    if token_budget < 1 or per_parent_budget < 1:
        raise ValueError('Token budgets must be positive')
    enc, parts = encoding(), []
    for rank, doc in enumerate(docs, 1):
        m = doc.metadata
        header = (f"[근거 {rank}] 파일: {m.get('file_name', m.get('source', ''))}\n"
                  f"섹션: {m.get('section_title', '')}\n"
                  f"매칭 child 페이지(0부터): {m.get('matched_child_pages', [])}\n")
        # Tokenize the full assembled string; separators/headers also count.
        prefix = '\n\n'.join(parts) + ('\n\n' if parts else '') + header
        available = min(per_parent_budget, token_budget - count_tokens(prefix) - 16)
        if available <= 0:
            break
        tokens = enc.encode(doc.page_content, disallowed_special=())
        anchor = m.get('matched_start_index', 0)
        if not isinstance(anchor, int) or not 0 <= anchor <= len(doc.page_content):
            anchor = 0
        # Keep the matched passage rather than blindly truncating the section's beginning.
        token_anchor = count_tokens(doc.page_content[:anchor])
        start = max(0, token_anchor - available // 4)
        start = min(start, max(0, len(tokens) - available))
        text = enc.decode(tokens[start:start + available])
        truncated = start > 0 or start + available < len(tokens)
        if truncated:
            text = '[관련 구간 발췌]\n' + text
        candidate = header + text
        assembled = '\n\n'.join(parts + [candidate])
        # Decoder boundaries can change tokenization; enforce the final exact count.
        while count_tokens(assembled) > token_budget and text:
            text_tokens = enc.encode(text, disallowed_special=())
            excess = count_tokens(assembled) - token_budget
            text = enc.decode(text_tokens[:max(0, len(text_tokens) - excess - 1)])
            candidate = header + text
            assembled = '\n\n'.join(parts + [candidate])
        if count_tokens(assembled) > token_budget:
            break
        parts.append(candidate)
        if debug:
            print(f"[{rank}] parent={m.get('parent_id')} 원문={len(tokens)}토큰 "
                  f"전달={count_tokens(candidate)}토큰 발췌={truncated}")
    context = '\n\n'.join(parts)
    if debug:
        print(f'컨텍스트: {count_tokens(context)}/{token_budget}토큰, 근거 {len(parts)}개')
    return context

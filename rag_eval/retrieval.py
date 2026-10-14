"""Offline ranking metrics. Duplicate evaluation IDs consume ranks but get no gain."""
import math


# 검색된 ID 목록과 정답 관련도 등급으로 질문 한 건의 검색 지표를 계산합니다.
def ranking_metrics(ids, relevance, ks=(1, 3, 5)):
    # 양수 등급의 정답이 있어야 Recall과 nDCG의 분모를 정의할 수 있습니다.
    if not relevance or not any(v > 0 for v in relevance.values()):
        raise ValueError('At least one positively graded relevant ID is required')
    # 등급 범위를 제한하고 비유한 값을 거부하여 이득 계산이 유효하도록 합니다.
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 20
           for v in relevance.values()):
        raise ValueError('Relevance grades must be finite numbers in [0, 20]')
    # 평가할 검색 깊이는 한 개 이상이어야 하며 양의 정수만 허용합니다.
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError('K must be a positive integer')
    # 중복 ID를 추적하면서 각 검색 순위가 얻을 관련도 등급을 기록합니다.
    seen, gains = set(), []
    for doc_id in ids:
        # 같은 ID가 다시 나오면 순위는 유지하지만 관련도 이득을 다시 주지 않습니다.
        grade = relevance.get(doc_id, 0) if doc_id not in seen else 0
        gains.append(grade)
        seen.add(doc_id)
    # 검색 결과의 개수가 아니라 정답 집합의 관련 ID 개수를 Recall 분모로 사용합니다.
    relevant_count = sum(v > 0 for v in relevance.values())
    # MRR은 저장된 순위에서 첫 관련 결과의 역순위이며 미검색이면 0점입니다.
    result = {'mrr': next((1 / rank for rank, g in enumerate(gains, 1) if g > 0), 0.0)}
    # 정답 등급을 내림차순으로 정렬하여 이상적인 검색 순서의 이득을 계산합니다.
    ideal = sorted(relevance.values(), reverse=True)
    for k in ks:
        # 등급이 높은 근거에 더 큰 이득을 주고 뒤쪽 순위는 로그로 할인합니다.
        dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))
        # 이상적인 상위 K개 이득을 계산하여 검색된 문서 수가 적어도 같은 기준을 사용합니다.
        idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
        # 상위 K개의 서로 다른 관련 ID 수를 전체 관련 ID 수로 나눕니다.
        result[f'recall@{k}'] = sum(g > 0 for g in gains[:k]) / relevant_count
        # 실제 할인 이득을 이상적인 할인 이득으로 나눠 순위 품질을 정규화합니다.
        result[f'ndcg@{k}'] = dcg / idcg
    return result

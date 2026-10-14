from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from .storage import validate_cases, write_json


# RAG 실행과 결과 수집을 담당하며 평가는 별도 함수에서 저장 결과로 수행합니다.
def collect(rag, cases, output, k=5, id_field='corpus_id'):
    # 유료 검색이나 생성 전에 질문과 기준 답변의 필수 입력을 확인합니다.
    validate_cases(cases)
    if k < 1 or id_field not in ('corpus_id', 'unit_id'):
        raise ValueError('Invalid K or evaluation ID field')
    # 질문과 정답 및 관련도 등급을 해시로 묶어 서로 다른 평가 데이터를 구별합니다.
    fingerprint = hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    # 검색 깊이와 시스템 설정 및 예상 질문 수를 기록하여 비교 조건을 검증할 수 있게 합니다.
    bundle = {'schema_version': 1, 'system': rag.name, 'config': rag.describe(),
              'dataset_sha256': fingerprint, 'retrieval_k': k, 'id_field': id_field,
              'created_at': datetime.now(timezone.utc).isoformat(), 'expected_count': len(cases), 'records': []}
    # Exclusive creation prevents accidentally overwriting a paid generation run.
    from pathlib import Path
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 배타적 생성 모드로 기존 수집 파일의 실수로 인한 덮어쓰기를 차단합니다.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(bundle, stream)
    for case in cases:
        # 각 질문의 검색을 한 번 수행하고 반환 순서를 그대로 보존합니다.
        docs = list(rag.retrieve(case['user_input'], k))
        # 어댑터가 요청한 K를 초과하면 비교 조건이 달라지므로 즉시 오류로 처리합니다.
        if len(docs) > k:
            raise ValueError('Adapter returned more than K documents')
        # 문서 ID와 검색 단위 ID 및 본문이 실제 문자열인지 검사합니다.
        for doc in docs:
            if not all(isinstance(v, str) and v.strip() for v in (doc.corpus_id, doc.unit_id, doc.text)):
                raise ValueError('Retrieved documents require nonempty IDs and text')
        # 질문과 이미 검색된 문서만 생성기에 전달하며 정답 정보는 전달하지 않습니다.
        generation = rag.generate(case['user_input'], docs)
        # 빈 답변이 정상 결과로 저장되어 평가에 포함되는 것을 막습니다.
        if not isinstance(generation.response, str) or not generation.response.strip():
            raise ValueError('Generator returned an empty response')
        # 실제 전달 맥락을 문자열 목록으로 보존할 수 있는지 확인합니다.
        if not isinstance(generation.contexts, list) or any(not isinstance(c, str) for c in generation.contexts):
            raise ValueError('contexts must be the list of strings actually sent to the generator')
        # 검색 원문과 실제 전달 맥락 및 답변을 함께 저장하여 두 평가를 분리합니다.
        bundle['records'].append({**case, 'response': generation.response,
                                  'retrieved_contexts': generation.contexts,
                                  'documents': [asdict(doc) for doc in docs]})
        # 질문 하나가 완료될 때마다 기록하여 후속 질문 실패 시 완료 결과를 보존합니다.
        write_json(output, bundle)
    return bundle


# 저장된 결과에만 접근하여 임베딩이나 답변 생성 없이 검색 품질을 평가합니다.
def evaluate_retrieval(bundle, ks):
    from .retrieval import ranking_metrics
    # 일부 질문만 수집된 실행은 평균 비교가 왜곡될 수 있어 거부합니다.
    if len(bundle['records']) != bundle['expected_count']:
        raise ValueError('Collection is incomplete; use only complete runs for comparison')
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError('K must be a positive integer')
    # 저장되지 않은 검색 순위를 추정하지 않도록 수집 깊이를 넘는 K를 거부합니다.
    if max(ks) > bundle['retrieval_k']:
        raise ValueError('Requested K exceeds the saved retrieval depth; collect at a larger K')
    rows = []
    for record in bundle['records']:
        # 수집 시 선택한 문서 단위 또는 검색 단위의 ID를 평가에 사용합니다.
        ids = [doc[bundle['id_field']] for doc in record['documents']]
        # 질문마다 독립적으로 검색 지표를 계산하고 질문 ID와 함께 저장합니다.
        rows.append({'case_id': record['case_id'], **ranking_metrics(ids, record.get('relevance'), ks)})
    if not rows:
        raise ValueError('No saved records')
    # 질문 식별자를 제외한 모든 검색 지표를 평균 계산 대상으로 선택합니다.
    names = [name for name in rows[0] if name != 'case_id']
    return {'system': bundle['system'], 'dataset_sha256': bundle['dataset_sha256'],
            'case_ids': [row['case_id'] for row in rows], 'id_field': bundle['id_field'],
            'retrieval_k': bundle['retrieval_k'], 'rows': rows,
            # 질문별 난이도와 문서 수에 관계없이 각 질문에 같은 가중치를 적용합니다.
            'summary': {name: sum(r[name] for r in rows) / len(rows) for name in names}}

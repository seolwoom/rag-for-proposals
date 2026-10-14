"""RAGAS scoring only consumes persisted artifacts; never calls a RAG."""
import asyncio
import math
import os
from importlib.metadata import version
from .storage import validate_cases, write_json


# 저장된 수집 결과를 채점하며 검색기와 답변 생성기는 호출하지 않습니다.
async def score_saved(bundle, output, model='gemini-3.1-flash-lite', timeout=300, metrics=None):
    records = bundle['records']
    # 모든 평가 질문의 수집이 끝났는지 확인하여 부분 실행의 비교를 막습니다.
    if len(records) != bundle['expected_count']:
        raise ValueError('Collection is incomplete')
    # 평가 API를 호출하기 전에 저장된 질문 및 기준 답변을 다시 검사합니다.
    validate_cases(records)
    # 각 기록에 생성 답변과 실제 전달 맥락이 올바른 자료형으로 저장됐는지 확인합니다.
    for record in records:
        if not isinstance(record.get('response'), str) or not record['response'].strip():
            raise ValueError('Saved response is required')
        contexts = record.get('retrieved_contexts')
        if not isinstance(contexts, list) or any(not isinstance(c, str) for c in contexts):
            raise ValueError('Saved contexts must be a list of strings')
    client = None
    # 실제 실행에서는 평가 모델을 만들고 테스트에서는 주입된 가짜 지표를 사용합니다.
    if metrics is None:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextRecall, Faithfulness, FactualCorrectness
        # 평가에 필요한 Google 키만 읽으므로 OpenAI 생성 키 없이도 재채점할 수 있습니다.
        key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not key:
            raise ValueError('Set GOOGLE_API_KEY or GEMINI_API_KEY; OpenAI generation key is not required')
        # Gemini의 OpenAI 호환 주소를 사용하고 요청 시간 제한과 재시도 횟수를 설정합니다.
        client = AsyncOpenAI(api_key=key, base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
                             timeout=120, max_retries=3)
        # 호환 클라이언트를 RAGAS 평가 모델로 감싸며 답변 생성 모델과 분리합니다.
        llm = llm_factory(model, provider='openai', client=client, max_tokens=4096)
        # 맥락 정보 충족도와 답변 근거성 및 기준 정답과의 사실 F1을 평가합니다.
        metrics = {'context_recall': ContextRecall(llm=llm), 'faithfulness': Faithfulness(llm=llm),
                   'factual_correctness': FactualCorrectness(llm=llm, mode='f1')}
        ragas_version = version('ragas')
    else:
        # 테스트 지표가 실제 RAGAS 버전의 결과로 오인되지 않도록 표시합니다.
        ragas_version = 'injected'
    # 평가 모델과 라이브러리 버전 및 데이터 조건을 결과에 기록하여 비교에 사용합니다.
    result = {'system': bundle['system'], 'dataset_sha256': bundle['dataset_sha256'],
              'retrieval_k': bundle['retrieval_k'], 'id_field': bundle['id_field'],
              'case_ids': [r['case_id'] for r in records], 'evaluator_model': model,
              'ragas_version': ragas_version, 'factual_correctness_mode': 'f1', 'rows': []}
    try:
        for record in records:
            row = {'case_id': record['case_id']}
            # 지표마다 필요한 입력만 전달하여 생성 답변과 기준 답변의 역할을 구분합니다.
            calls = {
                'context_recall': {k: record[k] for k in ('user_input', 'retrieved_contexts', 'reference')},
                'faithfulness': {k: record[k] for k in ('user_input', 'response', 'retrieved_contexts')},
                'factual_correctness': {k: record[k] for k in ('response', 'reference')},
            }
            for name, kwargs in calls.items():
                try:
                    # 한 지표의 채점이 오래 멈추지 않도록 비동기 호출에 제한 시간을 적용합니다.
                    score = await asyncio.wait_for(metrics[name].ascore(**kwargs), timeout=timeout)
                    value = float(score.value)
                    # NaN이나 무한대 및 범위를 벗어난 점수는 정상 평가값으로 저장하지 않습니다.
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError('Invalid score')
                    row[name], row[name + '_error'] = value, None
                # 한 지표 실패가 다른 지표의 채점을 중단하지 않도록 개별 오류를 처리합니다.
                except Exception as exc:
                    # 실패 점수는 null로 저장하고 민감한 요청 내용 대신 예외 유형만 기록합니다.
                    row[name], row[name + '_error'] = None, type(exc).__name__
            result['rows'].append(row)
            result['summary'] = {}
            for name in calls:
                # 실패를 0점으로 섞지 않고 성공한 점수만 평균에 반영합니다.
                values = [r[name] for r in result['rows'] if r[name] is not None]
                # 성공과 실패 건수를 함께 제공하여 부분 성공 평균을 해석할 수 있게 합니다.
                result['summary'][name] = {'mean': sum(values) / len(values) if values else None,
                                          'success': len(values), 'failure': len(result['rows']) - len(values)}
            # 질문별 채점이 끝날 때마다 결과와 현재 요약을 파일에 저장합니다.
            write_json(output, result)
    # 채점 중 예외가 발생하더라도 생성한 비동기 API 연결을 닫습니다.
    finally:
        if client is not None:
            await client.close()
    return result

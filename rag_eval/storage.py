import json
import os
from pathlib import Path


# UTF-8 JSON을 읽으며 Windows에서 붙을 수 있는 BOM도 허용합니다.
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


# 저장 디렉터리를 준비하고 임시 파일을 교체하여 결과 파일을 갱신합니다.
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 먼저 같은 디렉터리의 임시 파일에 기록하여 본 파일이 부분적으로 쓰이는 것을 줄입니다.
    temporary = path.with_suffix(path.suffix + '.tmp')
    # 한글을 그대로 기록하고 NaN을 금지하여 표준 JSON으로 읽을 수 있게 합니다.
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # 임시 파일 쓰기가 완료된 뒤 최종 경로의 파일을 교체합니다.
    os.replace(temporary, path)


# API 호출 전에 질문과 기준 답변 및 질문 식별자의 최소 요건을 검사합니다.
def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError('Cases must be a nonempty list')
    # 이미 확인한 질문 ID를 모아 질문별 결과가 서로 덮이거나 혼동되는 것을 막습니다.
    seen = set()
    for case in cases:
        # 질문 ID와 질문 및 기준 정답은 공백만 있는 문자열도 허용하지 않습니다.
        for key in ('case_id', 'user_input', 'reference'):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f'Missing nonempty {key}')
        # 질문 내용이 달라도 식별자가 같으면 비교할 수 없으므로 중복 ID를 거부합니다.
        if case['case_id'] in seen:
            raise ValueError('Duplicate case_id')
        seen.add(case['case_id'])

import argparse
import asyncio
import importlib
import json
from .storage import read_json, write_json


# 명령별로 필요한 모듈만 불러오는 공통 CLI 진입점입니다.
def main():
    parser = argparse.ArgumentParser(description='Collect once, evaluate saved RAG outputs independently')
    # 인덱스 구축과 수집 및 두 평가와 비교를 독립된 하위 명령으로 제공합니다.
    sub = parser.add_subparsers(dest='command', required=True)
    # 문서 폴더에서 새 인덱스를 만드는 명령의 입력 및 출력 경로를 정의합니다.
    build = sub.add_parser('build-index')
    build.add_argument('--files', default='files')
    build.add_argument('--index', default='faiss_index')
    # 사용자 어댑터와 평가 질문 및 저장 경로를 받아 결과 수집을 설정합니다.
    collect = sub.add_parser('collect')
    collect.add_argument('--adapter', required=True, help='module:zero_argument_factory')
    collect.add_argument('--cases', required=True)
    collect.add_argument('--output', required=True)
    collect.add_argument('--k', type=int, default=5)
    # 문서 단위 비교와 동일 청크 단위 비교 중 어떤 ID를 사용할지 선택합니다.
    collect.add_argument('--id-field', choices=['corpus_id', 'unit_id'], default='corpus_id')
    # 저장된 검색 결과의 평가 깊이 목록과 검색 평가 출력 경로를 정의합니다.
    retrieval = sub.add_parser('retrieval')
    retrieval.add_argument('--samples', required=True)
    retrieval.add_argument('--output', required=True)
    retrieval.add_argument('--ks', nargs='+', type=int, default=[1, 3, 5])
    # 저장된 답변을 채점할 평가 모델과 생성 평가 출력 경로를 정의합니다.
    ragas = sub.add_parser('ragas')
    ragas.add_argument('--samples', required=True)
    ragas.add_argument('--output', required=True)
    ragas.add_argument('--model', default='gemini-3.1-flash-lite')
    # 동일 조건에서 만들어진 여러 시스템의 평가 결과를 비교하도록 설정합니다.
    compare = sub.add_parser('compare')
    compare.add_argument('results', nargs='+')
    compare.add_argument('--output', required=True)
    args = parser.parse_args()
    # 외부 모델을 사용하는 명령에서만 환경 파일을 읽고 기존 환경변수를 우선합니다.
    if args.command in ('collect', 'build-index', 'ragas'):
        from dotenv import load_dotenv
        load_dotenv(encoding='utf-8-sig', override=False)
    # 인덱싱은 명시적으로 요청한 경우에만 실행하여 불필요한 임베딩 비용을 막습니다.
    if args.command == 'build-index':
        from .indexing import build_index
        build_index(args.files, args.index)
    elif args.command == 'collect':
        from .runner import collect as run
        # 모듈 경로와 인자 없는 생성 함수 이름을 분리하여 외부 팀 구현을 동적으로 연결합니다.
        module, factory = args.adapter.split(':')
        # Validate before constructing providers or touching the network.
        from .storage import validate_cases
        cases = read_json(args.cases)
        validate_cases(cases)
        # 지정한 factory가 만든 RAG 객체를 공통 수집 실행기에 전달합니다.
        run(getattr(importlib.import_module(module), factory)(), cases, args.output, args.k, args.id_field)
    # 검색 평가는 로컬 저장 파일만 읽으며 모델이나 API 키를 사용하지 않습니다.
    elif args.command == 'retrieval':
        from .runner import evaluate_retrieval
        write_json(args.output, evaluate_retrieval(read_json(args.samples), args.ks))
    # 비동기 채점 함수를 실행하며 저장된 답변과 맥락을 그대로 재사용합니다.
    elif args.command == 'ragas':
        from .ragas_scoring import score_saved
        asyncio.run(score_saved(read_json(args.samples), args.output, args.model))
    else:
        results = [read_json(p) for p in args.results]
        # Compare the same cases, annotations, retrieval depth and evaluator settings.
        # 데이터와 질문 집합 및 검색 깊이와 평가 모델 조건이 같아야 비교를 허용합니다.
        fields = ('dataset_sha256', 'case_ids', 'id_field', 'retrieval_k', 'evaluator_model',
                  'ragas_version', 'factual_correctness_mode')
        for other in results[1:]:
            if any(other.get(field) != results[0].get(field) for field in fields):
                raise ValueError('Comparison settings/datasets differ')
            # 평가 항목이 서로 다른 결과가 같은 비교 표에 들어오는 것을 막습니다.
            if set(other['summary']) != set(results[0]['summary']):
                raise ValueError('Comparison metrics differ')
        # 시스템 이름과 요약 지표를 한 행으로 묶어 비교용 JSON과 콘솔 출력에 사용합니다.
        table = [{'system': r['system'], **r['summary']} for r in results]
        write_json(args.output, table)
        print(json.dumps(table, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

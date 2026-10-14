# 팀원용 RAG 인터페이스 구현 안내

## 1. 구현 범위

자신의 RAG를 감싸는 **어댑터 객체**와 그 객체를 반환하는 **인자 없는 `create()` 함수**를 작성합니다. 기존 청킹·임베딩·벡터 DB·생성 모델은 그대로 사용해도 됩니다.

공통 자료형은 `rag_eval/contracts.py`에 정의되어 있습니다. `RAG` 클래스를 상속할 필요는 없습니다. 아래 메서드는 동기 함수로 작성합니다.

| 필수 항목 | 구현 내용 |
|---|---|
| `name` | 시스템을 구별하는 문자열 |
| `describe()` | 모델·인덱스 버전 등 JSON 저장 가능한 설정 딕셔너리 |
| `retrieve(question, k)` | 검색 순위대로 최대 K개의 `RetrievedDocument` 반환 |
| `generate(question, documents)` | 이미 검색된 문서로 답변을 생성하고 `Generation` 반환 |

## 2. 반환값 규칙

**`RetrievedDocument`**

- `corpus_id`: 공통 원문 ID. `files/기관A/사업.pdf`라면 `기관A/사업.pdf`를 사용합니다.
- `unit_id`: 반환한 청크 또는 parent의 ID. 예: `기관A/사업.pdf::chunk-001`.
- `text`: 검색 단위의 원문 텍스트.
- `score`: 선택적인 검색 점수. 없으면 `None`입니다.
- `metadata`: 페이지 등 JSON 저장 가능한 부가 정보입니다.

`corpus_id`, `unit_id`, `text`는 비어 있지 않은 문자열이어야 합니다. 모든 팀은 같은 원문에 같은 `corpus_id`를 사용하고, 평가셋의 `relevance` 키도 이 ID와 일치시킵니다. 파일명뿐 아니라 하위 경로·대소문자·문서 버전도 통일하세요.

**`Generation`**

- `response`: 비어 있지 않은 최종 답변 문자열.
- `contexts`: 생성 모델에 **실제로 전달한 근거 맥락**의 문자열 목록.

생성 시 다시 검색하지 않습니다. 맥락을 발췌하거나 합쳤다면 그 처리 후의 문자열을 반환합니다. 기준 답변과 relevance를 생성 입력에 넣지 않습니다.

## 3. 어댑터와 `create()` 예시

프로젝트 루트에 `team_rag.py`를 작성합니다. 아래의 `my_backend`는 팀원이 제공할 개인 모듈이며 현재 저장소에는 없습니다.

이 예시는 개인 백엔드가 `search(question, top_k)`로 순위순 딕셔너리 목록을, `answer(question, context)`로 답변 문자열을 반환한다고 가정합니다. 자신의 코드에 맞게 호출과 필드 변환을 수정하세요.

```python
from rag_eval import RetrievedDocument, Generation


# 기존 개인 RAG를 공통 반환 형식으로 연결합니다.
class TeamRAG:
    name = "team-a-v1"

    # 이미 준비된 검색 및 생성 백엔드를 보관합니다.
    def __init__(self, backend):
        self.backend = backend

    # 실제 설정을 기록하고 API 키나 모델 객체는 제외합니다.
    def describe(self):
        return {"embedding_model": "my-model", "index_version": "v1"}

    # 개인 검색 결과를 순위를 유지한 채 공통 자료형으로 변환합니다.
    def retrieve(self, question, k):
        hits = self.backend.search(question=question, top_k=k)
        return [
            RetrievedDocument(
                corpus_id=hit["relative_path"],
                unit_id=hit["unit_id"],
                text=hit["text"],
            )
            for hit in hits[:k]
        ]

    # 전달받은 문서로 생성하고 실제 전달 맥락을 함께 반환합니다.
    def generate(self, question, documents):
        context = "\n\n".join(doc.text for doc in documents)
        response = self.backend.answer(question=question, context=context)
        return Generation(response=response, contexts=[context] if context else [])


# 공통 CLI가 인자 없이 한 번 호출하는 객체 생성 함수입니다.
def create():
    from my_backend import load_backend

    # 개인 인덱스와 호환되는 질의 임베딩 및 생성기를 로드합니다.
    backend = load_backend(index_dir="team_index", embedding_model="my-model")
    return TeamRAG(backend)
```

`create()`는 **클래스가 아닌 초기화된 객체**를 반환합니다. 필요한 경로나 모델 설정은 함수 내부 또는 환경변수로 구성하세요. 반환 객체는 모든 평가 질문에 재사용되므로 인덱스를 매 질문마다 로드하지 않습니다.

## 4. 개인 임베딩 인덱스 사용

1. 자신의 인덱싱 코드로 인덱스를 미리 생성합니다.
2. 청크에 공통 `corpus_id`와 원문 텍스트를 보존합니다.
3. `create()`에서 개인 로더로 인덱스를 불러옵니다.
4. `retrieve()`에서 해당 인덱스와 호환되는 질의 임베딩을 사용합니다.

문서와 질문의 임베딩은 모델·버전·역할별 접두어·정규화 설정 등이 호환되어야 합니다. 벡터 차원이 같다는 것만으로 호환되지는 않습니다. 벡터와 원문 ID의 매핑도 정확히 복원해야 합니다.

현재 공통 `build-index`는 기존 OpenAI 구현만 실행합니다. 개인 인덱스를 현재 `OpenAIRAG`에 경로만 바꿔 넣지 말고, 위처럼 **자체 어댑터에서 로드**하세요. OpenAI로 다시 임베딩할 필요는 없습니다.

## 5. 공통 평가 실행

프로젝트 루트에서 실행합니다. 개인 모듈과 의존 패키지를 현재 Python 환경에서 불러올 수 있어야 합니다.

```powershell
# team_rag 모듈의 create()로 객체를 만들어 검색과 답변을 수집합니다.
python -m rag_eval collect --adapter team_rag:create --cases examples/proposal_cases.json --k 5 --output results/team-a/samples.json

# 저장 결과로 검색 지표를 계산합니다.
python -m rag_eval retrieval --samples results/team-a/samples.json --ks 1 3 5 --output results/team-a/retrieval.json

# 저장된 답변과 맥락으로 RAGAS를 실행합니다.
python -m rag_eval ragas --samples results/team-a/samples.json --output results/team-a/ragas.json
```

`team_rag:create`는 **모듈 이름:함수 이름**입니다. `.py` 확장자나 파일 경로를 넣지 않습니다. 수집 파일은 기존 파일을 덮어쓰지 않으므로 재수집할 때 새 출력 경로를 지정합니다.

팀 전체는 같은 평가셋·원문·K를 사용하세요. 서로 다른 청킹을 비교할 때는 기본 문서 단위 평가인 `corpus_id`를 사용합니다. 상세 설명은 [EVALUATION.md](EVALUATION.md)를 참고하세요.

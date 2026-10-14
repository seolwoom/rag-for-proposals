# 통합 RAG 평가 환경

v0.3 노트북을 `RAG_openai_ver_0_3_unified_eval_env.py`와 `rag_eval/`로 전환했습니다. 기존 노트북과 기존 청킹·리랭킹·맥락 모듈은 유지합니다. import만으로 인덱싱, 다운로드, API 호출을 실행하지 않습니다. 문서에 포함된 프롬프트는 생성 입력 데이터이며 개발 작업의 지시문이 아닙니다.

## 준비 및 실행

Python 3.10 이상에서 별도 가상환경을 만들고 아래 명령을 실행합니다.

```powershell
python -m pip install -r requirements-eval.txt
```

`.env`에는 생성·검색용 `OPENAI_API_KEY`, 평가용 `GOOGLE_API_KEY` 또는 `GEMINI_API_KEY`를 설정합니다. 환경변수가 `.env`보다 우선합니다. LangSmith를 사용할 때만 별도로 `LANGSMITH_TRACING=true`와 키를 설정합니다. RAGAS 재평가에는 OpenAI 키나 검색 인덱스가 필요하지 않습니다. 평가 시 저장된 질문·답변·맥락·기준 정답은 Gemini로 전송됩니다.

기존 v0.3 인덱스가 있으면 인덱싱을 생략합니다. 새로 만들 때:

```powershell
python -m rag_eval build-index --files files --index faiss_index
```

같은 저장 경로로 실행하면 기존 인덱스 파일을 덮어씁니다. 기존 결과를 보관하려면 `--index faiss_index_v2`처럼 별도 폴더를 지정합니다. 기본 dense·hybrid factory는 `faiss_index`를 읽으므로 새 경로를 쓰려면 아래 `create()` 예시처럼 어댑터에 경로를 지정해야 합니다. 개인 임베딩 인덱싱은 이 명령에 연결되어 있지 않으며 아래 개인 인덱스 절차를 참고하세요.

`examples/proposal_cases.json`에는 노트북의 질문·기준 정답 5개를 옮겼습니다. 정답을 원문과 대조하고 `relevance`에 관련 문서 ID와 등급을 채워야 검색 평가가 가능합니다. 정답 문서를 검색 결과에서 자동 추정하지 않습니다.

```json
[
  {
    "case_id": "q001",
    "user_input": "사업 수행 기한은?",
    "reference": "원문에서 확인한 기준 정답",
    "relevance": {"하위폴더/제안요청서.pdf": 2, "다른근거.hwp": 1}
  }
]
```

평가셋 파일 전체는 위처럼 질문 객체의 목록이어야 합니다. `case_id`는 중복 없이 지정하고 모든 시스템에 같은 파일을 사용합니다. `reference`는 RAGAS의 기준 답변이며 `relevance`는 검색 지표의 기준 문서 목록입니다. 관련도 기준은 예를 들어 0=무관, 1=보조 근거, 2=핵심 근거로 팀 전체에서 통일합니다. Recall과 MRR은 양수 여부만 사용하고 nDCG는 등급의 차이까지 반영합니다. 목록에 없는 ID는 관련도 0으로 취급하므로 실제 정답 근거를 빠뜨리지 않도록 원문을 확인하세요.

ID는 `files/`를 기준으로 한 상대 경로이며 `/`를 사용합니다. 대소문자·경로·코퍼스 버전을 팀 전체에서 통일해야 합니다. `faiss_index/parents.json`의 `metadata.relative_path`에서 현재 ID를 확인할 수 있습니다. 등급은 0~20의 유한 숫자이며 0은 무관, 양수는 관련입니다. `{}`는 미작성 상태이며 검색 평가에서 오류로 처리합니다. RAGAS는 문서 ID 정답 없이 기준 답변만으로 실행할 수 있습니다.

```powershell
# 질문당 검색과 답변 생성을 한 번 수행하고 즉시 저장
python -m rag_eval collect --adapter rag_eval.openai_rag:dense --cases examples/proposal_cases.json --k 5 --output results/dense/samples.json
python -m rag_eval collect --adapter rag_eval.openai_rag:hybrid --cases examples/proposal_cases.json --k 5 --output results/hybrid/samples.json

# 저장 데이터만 읽는 검색 지표 평가: API 호출 없음
python -m rag_eval retrieval --samples results/dense/samples.json --ks 1 3 5 --output results/dense/retrieval.json
python -m rag_eval retrieval --samples results/hybrid/samples.json --ks 1 3 5 --output results/hybrid/retrieval.json

# 저장 데이터만 읽는 RAGAS: 검색 및 답변 생성 없음
python -m rag_eval ragas --samples results/dense/samples.json --output results/dense/ragas.json
python -m rag_eval ragas --samples results/hybrid/samples.json --output results/hybrid/ragas.json

# 검색 / 생성 비교를 각각 실행
python -m rag_eval compare results/dense/retrieval.json results/hybrid/retrieval.json --output results/retrieval_comparison.json
python -m rag_eval compare results/dense/ragas.json results/hybrid/ragas.json --output results/ragas_comparison.json
```

`python RAG_openai_ver_0_3_unified_eval_env.py ...`도 동일한 명령을 지원합니다. 기존 노트북 기본값은 parent 4개였지만 위 비교는 @5를 평가하기 위해 5개를 요청합니다. 이 설정은 생성 맥락에도 영향을 주므로 팀 전체에서 같은 K를 사용하세요. 기존 4개 설정을 유지하려면 `--k 4 --ks 1 3 4`로 실행합니다. 저장 깊이보다 큰 K는 허용하지 않습니다.

## 팀원이 구현할 인터페이스

### 전체 연결 구조와 구현 범위

팀원은 기존 RAG를 그대로 개발하고, 공통 계약을 따르는 어댑터 객체와 그 객체를 만드는 `create()` 함수를 제공합니다. 어댑터는 기존 검색 결과와 답변을 공통 자료형으로 변환하는 연결 코드입니다. 상속, LangChain, OpenAI 사용은 필수가 아닙니다.

```text
각자의 인덱싱 코드 → 각자의 저장 인덱스
                            ↓ create()에서 로드
                     공통 계약을 구현한 객체
                            ↓
collect → retrieve(question, k) → generate(question, documents)
                            ↓
                       samples.json
                            ↓
               검색 지표 평가 / RAGAS 평가
```

인덱싱은 `contracts.py`에 포함되지 않습니다. 현재 `build-index` 명령은 이 프로젝트의 기존 OpenAI 인덱싱 코드만 호출합니다. 다른 팀원의 인덱싱 함수를 선택하는 옵션은 없습니다. 각자의 인덱싱을 먼저 실행하고 그 결과를 어댑터에서 읽으면 됩니다.

| 팀원에게 필요한 항목 | 역할 |
|---|---|
| 자체 RAG 코드와 의존 패키지 | 개인 임베딩, 벡터 DB, 검색, 생성 구현 |
| `rag_eval/contracts.py`의 자료형 | 검색 결과와 생성 결과 반환 |
| 어댑터 모듈 | 기존 RAG를 공통 계약에 연결 |
| 인자 없는 `create()` 함수 | 준비된 어댑터 객체를 공통 CLI에 반환 |
| 동일한 평가셋과 원문 ID 규칙 | 서로 다른 시스템을 동일한 정답으로 평가 |

공통 실행 환경에서는 `runner.py`, `storage.py`, `retrieval.py`, `ragas_scoring.py`, `__main__.py`, `__init__.py`를 사용합니다. 개인 구현은 `openai_rag.py`, `indexing.py`, `rag_chunking.py`, `rag_reranking.py`, `rag_context.py`에 의존할 필요가 없습니다. 공통 패키지의 재평가 명령은 개인 검색·생성 모듈을 로드하지 않습니다.

### 네 가지 필수 항목

| 항목 | 입력 | 출력 및 조건 |
|---|---|---|
| `name` | 없음 | 시스템을 구별하는 문자열 속성 |
| `describe()` | 없음 | JSON 저장 가능한 설정 딕셔너리 |
| `retrieve(question, k)` | 질문 문자열, 양의 정수 K | 관련도 순서대로 최대 K개의 `RetrievedDocument` |
| `generate(question, documents)` | 질문, 이미 검색된 문서 목록 | `Generation(response, contexts)` |

현재 실행기는 동기 메서드를 호출합니다. `async def retrieve()`나 `async def generate()`를 그대로 연결하면 동작하지 않습니다. 비동기 SDK를 쓰는 팀은 동기 어댑터를 제공해야 합니다. Protocol은 구조를 설명하는 타입 계약이며 실행기가 상속 여부를 검사하는 클래스가 아닙니다.

#### `name`과 `describe()`

`name`은 같은 팀에서도 검색 방식이나 모델 버전에 따라 구별되도록 지정합니다. 예를 들어 `team-a-local-dense-v1`과 `team-a-local-hybrid-v1`을 사용할 수 있습니다.

`describe()`에는 다음처럼 실제 설정을 기록합니다. 공통 평가기는 이 딕셔너리를 그대로 저장하지만 내용을 대신 검증하거나 모델을 설정하지 않습니다.

```python
# 현재 실행 조건을 기록하며 API 키나 모델 객체는 넣지 않습니다.
def describe(self):
    return {
        "embedding_model": "my-local-embedding-v1",
        "embedding_revision": "revision-20261014",
        "embedding_dimension": 768,
        "normalize_embeddings": True,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "vector_store": "my-store",
        "distance_metric": "cosine",
        "index_version": "corpus-v1-chunks-v2",
        "generator_model": "my-generator-v1",
        "prompt_version": "v2",
    }
```

`Path`, NumPy 배열, SDK 클라이언트, 모델 객체 등은 JSON에 직접 저장할 수 없습니다. 경로는 `str(path)`, 수치는 Python의 `int` 또는 `float`로 변환하세요. 키나 인증 정보는 저장하지 않습니다.

#### `retrieve(question, k)`

질문을 검색하고 최종적으로 생성에 전달할 검색 단위를 순위대로 반환합니다. 내부에서 K보다 많은 후보를 검색하고 리랭킹해도 되지만 반환 결과는 K개 이하로 제한합니다. 결과가 적으면 그대로 반환하고 같은 결과를 복제해 채우지 않습니다.

```python
from rag_eval import RetrievedDocument

# 개인 검색 결과를 공통 반환 형식으로 변환하는 예시입니다.
def retrieve(self, question, k):
    hits = self.backend.search(question=question, top_k=k)
    return [
        RetrievedDocument(
            corpus_id=hit["relative_path"],
            unit_id=hit["unit_id"],
            text=hit["text"],
            score=float(hit["score"]) if hit.get("score") is not None else None,
            metadata={"page": hit.get("page")},
        )
        for hit in hits[:k]
    ]
```

위 `backend.search()`와 결과 딕셔너리의 키는 설명을 위한 예시입니다. 실제 SDK의 반환값에 맞춰 변환해야 합니다. 검색 결과를 그대로 dict나 LangChain Document로 반환하는 대신 `RetrievedDocument`를 만들어 반환하세요. 실행기는 이를 `dataclasses.asdict()`로 저장합니다.

| `RetrievedDocument` 필드 | 의미와 작성 규칙 |
|---|---|
| `corpus_id` | 원본 문서의 공통 ID이며 모든 팀이 같은 원문에 같은 값을 사용 |
| `unit_id` | 실제 반환한 청크 또는 parent의 ID이며 같은 시스템과 인덱스에서 일관되게 유지 |
| `text` | 반환한 검색 단위의 원문 텍스트이며 빈 문자열은 허용하지 않음 |
| `score` | 선택적인 검색 점수이며 없는 경우 `None`, 있는 경우 유한한 Python 숫자 |
| `metadata` | 페이지, 출처, 청크 위치 등의 JSON 저장 가능한 부가 정보 |

`corpus_id`, `unit_id`, `text`는 모두 비어 있지 않은 문자열이어야 합니다. 벡터 자체를 `text`나 `metadata`에 넣을 필요는 없습니다. 거리 점수는 작을수록 좋을 수 있고 유사도는 클수록 좋을 수 있으므로 반환 목록의 순서를 실제 검색 순위로 맞추세요. 공통 평가기는 score로 재정렬하지 않습니다.

#### 원문 ID와 검색 단위 ID를 정하는 방법

`corpus_id`는 개별 청크가 아니라 원본 문서를 가리킵니다. 현재 팀 공통 규칙은 `files/` 기준 상대 경로입니다.

| 원본 파일 | `corpus_id` | 가능한 `unit_id` |
|---|---|---|
| `files/사업A.pdf` | `사업A.pdf` | `사업A.pdf::chunk-0001` |
| `files/기관B/사업A.pdf` | `기관B/사업A.pdf` | `기관B/사업A.pdf::chunk-0007` |
| `files/기관B/사업A.pdf`의 다른 청크 | `기관B/사업A.pdf` | `기관B/사업A.pdf::chunk-0008` |

모든 팀은 문서 내용과 버전, 파일명, 하위 폴더, 대소문자, 공백까지 통일해야 합니다. 경로 구분자는 `/`를 사용합니다. 파일 이름만 같고 하위 폴더가 다르면 현재 규칙에서는 다른 ID입니다. 로딩 시 상대 경로를 메타데이터에 넣고 청킹 후에도 유지하는 방식이 권장됩니다.

```python
from pathlib import Path

# 실제 파일의 절대 경로는 개인 환경마다 달라도 상대 경로는 같아야 합니다.
files_root = Path("files").resolve()
source_path = (files_root / "기관B" / "사업A.pdf").resolve()
corpus_id = source_path.relative_to(files_root).as_posix()
```

이미 만든 인덱스에 다른 ID가 들어 있다면 매핑 파일이나 변환 함수를 두세요. 같은 파일명이 여러 폴더에 있는 경우 이름만으로 매핑하면 충돌하므로 원본 위치를 기준으로 매핑해야 합니다.

```python
# 내부 DB 식별자를 평가셋의 공통 원문 ID로 바꾸는 예시입니다.
corpus_id_map = {
    "internal-doc-001": "기관B/사업A.pdf",
    "internal-doc-002": "기관C/사업A.pdf",
}
```

팀마다 청킹이 다르면 문서 ID 비교인 기본 `--id-field corpus_id`를 사용합니다. 같은 청킹·검색 단위 ID를 공유하는 경우에만 `--id-field unit_id`와 그 ID 기준의 relevance를 사용하세요.

#### `generate(question, documents)`와 실제 맥락

이미 검색된 문서로 답변을 생성해야 합니다. 검색과 생성을 함께 수행하는 기존 `rag_chain.invoke(question)`를 그대로 호출하면 내부에서 다시 검색될 수 있습니다. 생성 전용 함수를 분리하거나 질문과 맥락을 직접 전달할 수 있는 경로를 사용하세요.

```python
from rag_eval import Generation

# 모델에 실제 전달한 근거 문자열을 답변과 함께 반환합니다.
def generate(self, question, documents):
    contexts = [
        f"[근거 {rank}] {doc.corpus_id}\n{doc.text[:1500]}"
        for rank, doc in enumerate(documents, start=1)
    ]
    context = "\n\n".join(contexts)
    response = self.backend.answer(question=question, context=context)
    return Generation(response=str(response), contexts=[context] if context else [])
```

이 예시의 1,500자 발췌는 설명용 설정이며 토큰 제한과 동일하지 않습니다. 실제 구현에서는 자신의 발췌 방식과 모델 입력 한도에 맞게 작성하고 설정을 `describe()`에 기록하세요. 합친 맥락 전체를 모델에 전달했다면 `[context]`로 보존하고, 별도 근거 블록들을 전달했다면 그 블록 목록을 보존할 수 있습니다. RAGAS에 시스템 지시문과 질문 전체를 맥락으로 넣지 말고 답변 근거로 전달된 내용을 넣습니다.

`Generation.response`는 비어 있지 않은 최종 답변 문자열이고 `contexts`는 문자열 목록입니다. 근거를 찾지 못한 질문에서는 `documents=[]`, `contexts=[]`로 처리하고 답변 불가 등의 비어 있지 않은 답변을 반환할 수 있습니다. 빈 맥락의 RAGAS 채점은 지표 구현에 따라 실패할 수 있으며 실패 건수를 확인해야 합니다. 평가셋의 `reference`와 `relevance`는 생성 메서드에 전달되지 않습니다.

### `def create()` 작성 방법

`create()`는 CLI가 인자 없이 한 번 호출하는 함수입니다. 검색이나 인덱싱 결과 자체를 반환하는 함수가 아니라 `name`, `describe()`, `retrieve()`, `generate()`를 가진 준비된 객체를 반환합니다. 해당 객체는 한 번의 collect 실행에 포함된 모든 질문에 재사용됩니다.

`--adapter team_rag:create`는 다음 순서로 처리됩니다.

1. Python에서 `team_rag` 모듈을 import합니다.
2. 모듈 안의 `create` 함수를 찾습니다.
3. 인자 없이 `create()`를 호출합니다.
4. 반환된 객체의 설정을 읽고 질문마다 검색과 생성을 호출합니다.

#### 예시 1: 기존 OpenAI 어댑터의 경로만 변경

프로젝트 루트에 `team_openai.py`를 만들고 아래 내용을 넣습니다. 같은 방식으로 생성한 v0.3 인덱스를 다른 폴더에서 읽는 경우에 사용합니다.

```python
from rag_eval.openai_rag import OpenAIRAG

# 기존 OpenAI 어댑터를 다른 인덱스 경로로 초기화합니다.
def create():
    return OpenAIRAG(
        index_dir="faiss_index_v2",
        mode="hybrid",
        candidate_k=20,
    )
```

```powershell
python -m rag_eval collect --adapter team_openai:create --cases examples/proposal_cases.json --k 5 --output results/team-openai/samples.json
```

이 예시는 OpenAI 어댑터가 지원하는 인덱스에만 해당합니다. 개인 임베딩 인덱스는 아래의 자체 어댑터 예시를 사용하세요.

#### 예시 2: 개인 검색·생성 코드를 감싸는 어댑터

프로젝트 루트의 `team_rag.py`에 다음 구조로 작성합니다. `TeamAdapter`는 전체 인터페이스 예시이고, `my_rag_backend`는 팀원이 직접 제공할 모듈입니다. 이 이름의 모듈은 현재 저장소에 없으므로 그대로 복사하는 것만으로 개인 RAG가 실행되지는 않습니다.

```python
import os
from pathlib import Path
from rag_eval import RetrievedDocument, Generation

# 개인 백엔드의 반환값을 공통 평가 계약으로 변환합니다.
class TeamAdapter:
    name = "team-a-personal-embedding-v1"

    # 준비된 검색 및 생성 백엔드를 보관합니다.
    def __init__(self, backend, index_dir, embedding_model):
        self.backend = backend
        self.index_dir = index_dir
        self.embedding_model = embedding_model

    # 실제 실행 설정을 기록하며 키나 SDK 객체는 저장하지 않습니다.
    def describe(self):
        return {
            "embedding_model": self.embedding_model,
            "index_dir": str(self.index_dir),
            "index_version": "v1",
            "prompt_version": "v1",
        }

    # 백엔드는 내부적으로 같은 개인 임베딩 모델로 질문을 벡터화해야 합니다.
    def retrieve(self, question, k):
        hits = self.backend.search(question=question, top_k=k)
        documents = []
        for hit in hits[:k]:
            score = hit.get("score")
            documents.append(RetrievedDocument(
                corpus_id=hit["relative_path"],
                unit_id=hit["unit_id"],
                text=hit["text"],
                score=float(score) if score is not None else None,
                metadata={"page": hit.get("page")},
            ))
        return documents

    # 이미 검색한 문서로 답변을 생성하며 내부에서 다시 검색하지 않습니다.
    def generate(self, question, documents):
        contexts = [
            f"[근거 {rank}] {doc.corpus_id}\n{doc.text}"
            for rank, doc in enumerate(documents, start=1)
        ]
        context = "\n\n".join(contexts)
        response = self.backend.answer(question=question, context=context)
        return Generation(
            response=str(response),
            contexts=[context] if context else [],
        )

# CLI는 인자를 전달하지 않으므로 필요한 설정은 함수 내부에서 구성합니다.
def create():
    from my_rag_backend import load_backend

    index_dir = Path(os.getenv("TEAM_INDEX_DIR", "team_index")).resolve()
    embedding_model = os.getenv("TEAM_EMBEDDING_MODEL", "my-local-model-v1")
    if not index_dir.is_dir():
        raise FileNotFoundError(f"개인 인덱스를 먼저 생성하세요: {index_dir}")

    # 개인 코드가 구현한 로더로 기존 인덱스와 질의 임베딩 및 생성기를 준비합니다.
    backend = load_backend(
        index_dir=index_dir,
        embedding_model=embedding_model,
    )
    return TeamAdapter(backend, index_dir, embedding_model)
```

예시의 백엔드는 `search(question, top_k)`가 순위순 딕셔너리 목록을 반환하고 `answer(question, context)`가 답변 문자열을 반환한다고 가정합니다. 자신의 백엔드가 Document, 튜플, SDK 응답 등을 반환한다면 변환 코드만 맞추면 됩니다. 모델 응답 객체는 그대로 `str()`으로 만들기보다 실제 답변 필드인 `content` 또는 `text` 등을 추출해야 합니다.

```powershell
$env:TEAM_INDEX_DIR = "team_index"
$env:TEAM_EMBEDDING_MODEL = "my-local-model-v1"
python -m rag_eval collect --adapter team_rag:create --cases examples/proposal_cases.json --k 5 --output results/team-a/samples.json
```

CLI는 `.env`도 읽지만 이미 설정된 환경변수가 우선합니다. `create()`에서 기존 인덱스를 로드하고 모델을 준비하되 매번 전체 문서를 다시 임베딩하지 마세요. 인덱스가 없다면 별도 인덱싱 명령을 실행하도록 오류를 안내하는 방식이 좋습니다.

#### 함수 이름, 파일 위치, 자주 하는 실수

`create`는 필수 이름이 아닙니다. 인자 없는 함수라면 `make_rag()`를 제공하고 `--adapter team_rag:make_rag`로 지정할 수 있습니다. 현재 CLI에는 factory에 임의 인자를 전달하는 옵션이 없습니다. `create(index_dir)`처럼 필수 인자를 요구하지 말고 환경변수나 함수 내부 설정으로 해결하세요.

`team_rag.py`가 프로젝트 루트에 있으면 `team_rag:create`, `adapters/team_a.py`가 패키지로 구성되어 있으면 `adapters.team_a:create`를 사용합니다. 패키지 구성 시 `adapters/__init__.py`를 두면 경로를 이해하기 쉽습니다. `--adapter`에는 `team_rag.py:create`처럼 확장자나 파일 경로를 넣지 않습니다. 외부 저장소의 구현은 현재 Python 환경에서 import할 수 있도록 설치하거나 모듈 경로를 준비해야 합니다.

| 잘못된 작성 | 수정 방법 |
|---|---|
| `return TeamAdapter` | 클래스가 아닌 객체를 `return TeamAdapter(...)`로 반환 |
| `return backend.search(...)` | 결과 목록 대신 계약을 구현한 객체를 반환 |
| `def create(index_dir): ...` | 인자 없는 함수 내부에서 경로를 구성 |
| 함수 마지막에 `return` 없음 | 생성한 객체를 명시적으로 반환 |
| 매 질문마다 인덱스 구축 | 인덱싱은 별도 실행하고 factory에서 한 번 로드 |
| 생성 함수가 다시 검색 | 이미 전달된 documents로 생성 |
| 전체 RAG chain에서 답변만 추출 | 검색 결과와 실제 생성 맥락도 별도로 반환 |

`examples/custom_rag.py`는 연결 구조를 확인하는 고정 답변 예제이며 실제 품질 비교용 RAG가 아닙니다. 자신의 구현과 연결하기 전에 참고용으로 사용할 수 있습니다.

## 개인 임베딩으로 만든 인덱스를 사용하는 방법

### 인덱싱과 공통 평가는 분리되어 있습니다

개인 임베딩으로 만든 인덱스를 다시 OpenAI로 임베딩할 필요는 없습니다. 자체 인덱스와 자체 질의 임베딩을 사용하는 어댑터를 만들고 공통 반환 형식으로 결과를 전달하면 됩니다. 로컬 모델, 다른 API, 자체 학습 모델, FAISS, Chroma, 원격 벡터 DB 등 어떤 내부 방식이든 공통 계약을 만족하면 평가 경로는 같습니다.

| 단계 | 누가 담당하는가 | 반드시 맞출 조건 |
|---|---|---|
| 문서 추출과 청킹 | 팀원의 개인 코드 | 공통 원문 ID를 각 검색 단위에 보존 |
| 문서 임베딩과 인덱스 저장 | 팀원의 개인 코드 | 모델과 전처리 및 벡터 매핑 기록 |
| 인덱스 로드 | `create()`에서 개인 로더 호출 | 자신이 저장한 형식으로 로드 |
| 질문 임베딩과 검색 | 어댑터의 `retrieve()` | 해당 인덱스와 호환되는 질의 인코딩 |
| 답변 생성 | 어댑터의 `generate()` | 전달받은 문서 사용 및 실제 맥락 반환 |
| 지표 계산 | 공통 평가 코드 | 동일한 평가셋과 ID 및 평가 조건 사용 |

### 문서 벡터와 질문 벡터의 호환성이 중요합니다

문서를 모델 A로 임베딩했다면 검색 질문도 그 모델과 호환되는 질의 인코더를 사용해야 합니다. 벡터 차원이 같다고 검색 공간이 같은 것은 아닙니다. 일부 모델은 문서와 질의에 서로 다른 접두어나 역할별 인코딩을 요구하므로 각 역할에 맞는 설정을 유지합니다.

개인 인덱스의 설정 파일에는 다음 정보를 보관하고 `create()`에서 같은 조건을 복원하세요.

- 임베딩 모델 이름과 버전 또는 체크포인트.
- 문서 및 질의 인코딩 방식과 접두어.
- 벡터 차원과 정규화 여부.
- cosine, inner product, L2 등 거리 계산 방식.
- 문서 추출과 청킹 버전 및 코퍼스 버전.
- 벡터 행 또는 DB ID와 원문 텍스트·corpus_id·unit_id의 매핑.

예를 들어 인덱싱 때 정규화한 벡터에 내적 검색을 사용했다면 질의에도 같은 정규화 조건을 적용해야 합니다. 개인 로더에서 설정을 읽고 호환성을 확인하세요. 모델 변경 시에는 문서 벡터를 새로 만들고 질의 모델도 같이 변경해야 합니다.

### 현재 OpenAI 어댑터에 개인 인덱스를 넣으면 안 되는 이유

현재 `OpenAIRAG`는 manifest의 모델 이름으로 `OpenAIEmbeddings`를 생성하고, v0.3의 청킹 방식과 파일 구조를 전제로 로드합니다. 로컬 임베딩 객체를 전달하는 생성자 인자는 없습니다.

`manifest.json`의 `embedding_model`을 개인 모델 이름으로 바꾸거나 인덱스 경로만 변경하는 것으로 개인 임베딩 지원이 되지 않습니다. OpenAI로 질문을 벡터화하려 하거나 벡터 공간이 일치하지 않는 검색이 발생할 수 있습니다. `create()`에서 개인 임베딩 모델과 개인 벡터 저장소를 사용하는 `TeamAdapter`를 반환하세요.

FAISS를 쓰더라도 외부에서 만든 벡터의 행 순서와 문서 목록의 순서가 v0.3과 같다는 보장이 없습니다. 개인 인덱스가 저장한 실제 ID 매핑을 복원해야 합니다. 임의로 chunks.json의 순서와 연결하면 다른 문서의 원문을 반환할 수 있습니다. 원격 DB라면 인덱스 디렉터리 대신 collection 이름이나 접속 설정을 factory에서 구성해도 됩니다.

### 개인 인덱스의 전체 실행 순서

아래 첫 명령의 `my_indexing.py`는 팀원이 작성한 파일을 뜻하며 공통 저장소에서 제공하는 명령이 아닙니다.

```powershell
# 개인 코드로 같은 참고문서를 추출하고 개인 임베딩 인덱스를 생성합니다.
python my_indexing.py

# 개인 인덱스를 읽는 어댑터를 선택하고 검색 및 답변을 한 번 수집합니다.
python -m rag_eval collect --adapter team_rag:create --cases examples/proposal_cases.json --k 5 --output results/team-a/samples.json

# 이후 검색 지표는 모델이나 개인 인덱스 없이 저장 결과로 계산합니다.
python -m rag_eval retrieval --samples results/team-a/samples.json --ks 1 3 5 --output results/team-a/retrieval.json

# 이후 RAGAS는 답변 재생성 없이 저장 결과를 Gemini로 채점합니다.
python -m rag_eval ragas --samples results/team-a/samples.json --output results/team-a/ragas.json
```

개인 임베딩과 생성 SDK의 패키지는 각 팀원이 추가로 설치해야 합니다. 공통 requirements 파일이 모든 개인 구현의 패키지와 버전까지 보장하지 않습니다. 인덱싱용 개인 키와 생성용 개인 키는 팀원 코드에서 읽고, RAGAS 채점용 Google 키는 공통 채점 코드에서 읽습니다.

인덱스를 수정하거나 임베딩 모델을 변경했으면 새 출력 경로로 다시 collect해야 새 검색과 답변을 평가할 수 있습니다. 기존 samples.json은 이전 실행의 스냅샷이며 인덱스 변경을 자동 반영하지 않습니다. 반대로 RAGAS만 재실행할 때는 개인 인덱스나 임베딩 모델을 준비할 필요가 없습니다.

### 팀원 연결을 확인하는 최소 점검

연결 후 전체 평가 전에 질문 한 건으로 반환 구조를 확인할 수 있습니다. 아래 코드는 개인 검색 및 생성 API를 실제 호출할 수 있습니다.

```python
import json
from dataclasses import asdict
from team_rag import create

# 평가 실행기와 동일하게 factory를 인자 없이 호출합니다.
rag = create()
assert isinstance(rag.name, str) and rag.name
json.dumps(rag.describe(), ensure_ascii=False, allow_nan=False)

# 실제 참고문서에 맞는 테스트 질문으로 바꿉니다.
question = "사업 수행 기한은 언제인가요?"
documents = list(rag.retrieve(question, 3))
assert len(documents) <= 3
for document in documents:
    assert document.corpus_id.strip()
    assert document.unit_id.strip()
    assert document.text.strip()
    json.dumps(asdict(document), ensure_ascii=False, allow_nan=False)

# 이미 검색된 문서로 생성하며 실제 전달 맥락을 반환하는지 확인합니다.
generation = rag.generate(question, documents)
assert isinstance(generation.response, str) and generation.response.strip()
assert isinstance(generation.contexts, list)
assert all(isinstance(context, str) for context in generation.contexts)
print([document.corpus_id for document in documents])
print(generation.response)
```

타입과 JSON 검사가 성공해도 ID가 정답 relevance와 일치하는지, 질의 임베딩이 개인 인덱스와 맞는지, 생성 시 재검색하지 않는지, 실제 전달 맥락을 저장하는지는 직접 확인해야 합니다.

## 지표의 정의와 비교 조건

- Recall@K: 상위 K개에서 찾은 서로 다른 관련 ID 수 / 전체 관련 ID 수.
- MRR: **저장된 전체 검색 깊이 내** 첫 관련 결과 순위의 역수. 찾지 못하면 0. 무제한 코퍼스 전체에 대한 MRR이 아니므로 저장 깊이를 통일합니다.
- nDCG@K: `(2^grade - 1) / log2(rank + 1)`의 합을 이상적인 상위 K개 순위 합으로 나눕니다.
- 중복 ID는 순위 자리를 차지하되 두 번째부터 관련 gain을 얻지 않습니다. 문서 단위 비교에서 여러 parent가 같은 문서에 속하는 경우에도 적용합니다.
- 모든 질문을 같은 가중치로 평균합니다. 검색 정답이 없는 질문은 0점으로 취급하지 않고 입력 오류로 처리합니다.
- 기본 비교 단위는 문서 `corpus_id`입니다. 서로 다른 청킹 결과를 비교할 때 적합합니다. 동일 청킹을 사용하는 팀은 `collect --id-field unit_id`와 해당 ID relevance로 검색 단위 비교를 할 수 있습니다. 현재 parent ID는 코퍼스 순서에 의존하므로 다른 인덱스 버전에서 동일 ID로 간주하면 안 됩니다.
- `context_recall`은 기준 답변에 필요한 정보가 실제 전달 맥락에 얼마나 포함되는지, `faithfulness`는 생성 답변이 전달 맥락에 근거하는지, `factual_correctness`는 기준 답변과 생성 답변의 사실 일치도를 F1 모드로 평가합니다. [RAGAS 공식 지표 문서](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/).

RAGAS 0.4.3 collections API를 사용합니다. 질문별 metric 실패는 JSON `null`과 예외 유형으로 저장합니다. 평균에서 실패를 제외하며 성공·실패 건수를 함께 표시합니다. 실패 건수가 다른 시스템의 평균을 단독으로 우열 판정에 쓰지 마세요. 평가 모델·버전·데이터·질문 집합이 다른 결과를 `compare`에 넣으면 오류가 발생합니다. RAGAS의 확률적 채점은 반복 실행 시 달라질 수 있습니다.

개인 임베딩·생성 모델·검색 방식 등 비교하려는 시스템 설정은 서로 달라도 됩니다. 현재 compare는 describe의 설정 전체를 같도록 강제하지 않습니다. 팀이 통제할 조건과 비교할 조건을 미리 정하고 각 시스템 설정을 저장된 samples.json에서 확인하세요. 공통 정답 데이터가 같아도 실제 코퍼스 파일 내용이 같은지 자동으로 검사하지 않으므로 동일 원문을 공유해야 합니다.

`RateLimitError`가 발생하면 평가 API의 요청량이나 토큰 사용량 및 할당량을 확인하세요. 현재 구현은 순차 채점과 클라이언트 재시도 3회를 사용하지만 별도의 호출 간격 제한이나 실패 항목만 골라 재채점하는 기능은 없습니다. 제한이 해소된 뒤 ragas 명령을 재실행하면 저장된 답변을 재사용하되 성공했던 항목도 다시 채점합니다.

## 저장과 검증

`samples.json`에는 스키마 버전, 시스템 설정, 데이터 SHA256, K, 평가 ID 단위, 질문별 전체 검색 결과/메타데이터/점수, 실제 전달 맥락, 생성 답변, 기준 정답을 저장합니다. 매 질문 완료 후 임시 파일을 교체하여 기록합니다. 중간 실패 시 완료된 질문은 보존하지만 불완전한 수집은 평가하지 않습니다. 같은 출력 파일로 재수집하면 덮어쓰기를 거부합니다. 자동 이어서 수집은 제공하지 않으며 새 출력 경로로 재실행하세요. 저장된 전체 결과는 평가 재실행 시 검색·생성 비용을 발생시키지 않습니다.

OpenAI 어댑터는 v0.3 `FAISS.from_documents`가 chunks.json 순서로 벡터를 추가한 인덱스를 대상으로 JSON docstore를 재구성합니다. 외부에서 다른 순서로 만든 FAISS 인덱스는 별도 어댑터를 사용하세요. `index.pkl`을 역직렬화하지 않습니다.

```powershell
python -m unittest test_unified_eval -v
python -m compileall -q rag_eval examples/custom_rag.py RAG_openai_ver_0_3_unified_eval_env.py
```

테스트는 등급형 nDCG, 중복 검색, 빈 검색, 저장 깊이 검증, 결과 덮어쓰기 방지, 실제 맥락 전달, metric 실패 집계, 반복 RAGAS 평가 시 검색·생성 호출이 늘어나지 않는지 검증합니다. 실제 OpenAI/Gemini 호출 및 문서 인덱스 재생성은 별도 환경·키를 갖춘 뒤 확인해야 합니다.

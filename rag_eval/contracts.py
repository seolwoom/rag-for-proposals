from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
# 검색 순서대로 반환할 문서 한 건을 표현하며 특정 RAG 라이브러리에 의존하지 않습니다.
class RetrievedDocument:
    # corpus_id must identify the same source across all participating systems.
    # 모든 팀이 같은 원문에 같은 ID를 사용하며 일반적으로 코퍼스 기준 상대 경로를 넣습니다.
    corpus_id: str
    # 검색된 단위의 원문 텍스트를 저장하여 검색 근거를 나중에도 확인할 수 있게 합니다.
    text: str
    # 문서 안의 parent 또는 청크를 구별하며 동일한 청킹 체계에서 세부 검색 비교에 사용합니다.
    unit_id: str
    # 검색기가 제공한 점수를 선택적으로 저장하며 서로 다른 시스템의 점수를 직접 비교하지 않습니다.
    score: float | None = None
    # 출처와 페이지 등 부가 정보를 보존하며 문서마다 독립적인 딕셔너리를 생성합니다.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
# 생성 답변과 실제 생성 입력 맥락을 함께 반환하여 재생성 없이 평가할 수 있게 합니다.
class Generation:
    # 모델이 생성한 최종 답변 문자열을 저장하며 기준 정답과 구분합니다.
    response: str
    # Exact context strings supplied to the generator, including truncation.
    # 포맷과 발췌가 적용된 실제 전달 문자열을 반환하며 검색 원문 전체로 대체하지 않습니다.
    contexts: list[str]


# 이 메서드 구성을 갖춘 객체는 상속 없이도 공통 평가 실행기에 연결할 수 있습니다.
class RAG(Protocol):
    # 비교 결과에서 시스템을 식별할 수 있도록 팀별로 구별되는 이름을 지정합니다.
    name: str

    # 모델과 검색 설정 등 실행 조건을 JSON으로 저장 가능한 형태로 반환합니다.
    def describe(self) -> dict[str, Any]: ...

    # 질문과 검색 깊이를 받아 상위 K개 이내의 문서를 관련도 순서대로 반환합니다.
    def retrieve(self, question: str, k: int) -> Sequence[RetrievedDocument]: ...

    # 이미 검색한 문서만 사용하여 답변을 생성하며 내부에서 다시 검색하지 않아야 합니다.
    def generate(self, question: str, documents: Sequence[RetrievedDocument]) -> Generation: ...

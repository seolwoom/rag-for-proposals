"""Replace these bodies with your team's RAG. No evaluator dependency required."""
from rag_eval import RetrievedDocument, Generation


# 팀별 RAG를 연결할 최소 예제이며 실제 검색과 생성 코드는 각 메서드를 교체합니다.
class TeamRAG:
    # 다른 팀의 결과와 구별되도록 시스템 이름을 정합니다.
    name = 'team-example'

    # 실제 구현에서는 모델과 프롬프트 버전 및 검색 설정을 함께 반환합니다.
    def describe(self):
        return {'revision': 'example-only'}

    # 질문에 맞는 근거를 순위대로 반환하고 요청한 K개를 초과하지 않게 합니다.
    def retrieve(self, question, k):
        # corpus_id is the agreed corpus-relative path, NOT an absolute local path.
        return [RetrievedDocument('example.pdf', '예제 근거', 'example.pdf::chunk-1')][:k]

    # 이미 받은 문서로 답변을 만들고 실제로 사용한 맥락을 함께 반환합니다.
    def generate(self, question, documents):
        # 이 예제는 원문을 그대로 사용하지만 실제 구현은 발췌 후 전달 문자열을 기록합니다.
        contexts = [doc.text for doc in documents]
        # Return exactly what was passed into your generator after all formatting.
        return Generation('예제 답변', contexts)


# 명령줄의 module:factory 형식에서 호출할 인자 없는 진입 함수입니다.
def create():
    return TeamRAG()

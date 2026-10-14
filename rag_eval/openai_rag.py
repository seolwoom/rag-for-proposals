"""Notebook dense/hybrid pipeline adapted to the shared contract."""
import hashlib
from pathlib import Path
from .contracts import RetrievedDocument, Generation
from .storage import read_json


# 노트북의 dense 및 hybrid 파이프라인을 공통 인터페이스에 맞춘 어댑터입니다.
class OpenAIRAG:
    # 저장된 인덱스를 읽고 검색 및 생성 구성 요소를 준비하며 새 임베딩은 만들지 않습니다.
    def __init__(self, index_dir='faiss_index', mode='dense', candidate_k=20):
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_community.vectorstores import FAISS
        from langchain_community.docstore.in_memory import InMemoryDocstore
        import faiss
        from rag_context import LLM_MODEL, OUTPUT_TOKEN_LIMIT
        if mode not in ('dense', 'hybrid') or candidate_k < 1:
            raise ValueError('Invalid mode/candidate_k')
        # 검색 방식과 후보 개수를 보관하여 실제 실행 조건을 결과 설정에 남깁니다.
        self.mode, self.candidate_k = mode, candidate_k
        self.name = f'openai-{mode}'
        directory = Path(index_dir)
        # 인덱스의 청킹 방식과 임베딩 설정을 읽어 현재 구현과 호환되는지 확인합니다.
        self.manifest = read_json(directory / 'manifest.json')
        if self.manifest.get('chunking_strategy') != 'section_parent_child_v2':
            raise ValueError('Rebuild the index using section_parent_child_v2')
        # JSON의 parent 원문을 복원하여 검색된 child 주변의 맥락을 생성에 사용할 수 있게 합니다.
        self.parents = {key: Document(**row) for key, row in read_json(directory / 'parents.json').items()}
        # 저장된 child 목록을 원래 순서대로 복원하여 벡터 행과 연결합니다.
        self.children = [Document(**row) for row in read_json(directory / 'chunks.json')]
        # 벡터 인덱스만 읽으며 Python 객체가 들어 있는 pickle 파일은 로드하지 않습니다.
        index = faiss.read_index(str(directory / 'index.faiss'))
        # 벡터 개수 및 차원이 저장 문서와 설정에 맞는지 검사합니다.
        if index.ntotal != len(self.children) or index.d != self.manifest['embedding_dimension']:
            raise ValueError('Index/chunks/manifest mismatch')
        # Notebook FAISS.from_documents adds vectors in chunks.json order.
        # Reconstruct the docstore from JSON without loading index.pkl.
        # 노트북이 child를 추가한 순서대로 벡터 행 번호와 JSON 문서를 연결합니다.
        mapping = {i: str(i) for i in range(len(self.children))}
        # 복원된 docstore와 벡터 인덱스로 질의 임베딩 기반 검색기를 만듭니다.
        self.store = FAISS(OpenAIEmbeddings(model=self.manifest['embedding_model']), index,
                           InMemoryDocstore({str(i): d for i, d in enumerate(self.children)}), mapping)
        # 기존 생성 모델과 출력 한도 및 추론 설정을 유지합니다.
        self.llm = ChatOpenAI(model=LLM_MODEL, max_completion_tokens=OUTPUT_TOKEN_LIMIT, reasoning_effort='low')
        # 노트북에서 추출한 생성 프롬프트를 별도 파일에서 읽습니다.
        self.prompt = Path(__file__).with_name('prompt.txt').read_text(encoding='utf-8')
        self.bm25 = None
        # hybrid 방식에서만 BM25를 구성하여 dense 실행의 불필요한 준비를 줄입니다.
        if mode == 'hybrid':
            from langchain_community.retrievers import BM25Retriever
            # 소문자 및 공백 기반 토큰화로 기존 노트북과 같은 키워드 검색을 구성합니다.
            self.bm25 = BM25Retriever.from_documents(self.children, preprocess_func=lambda text: text.lower().split())
            self.bm25.k = candidate_k

    # 모델과 맥락 예산 및 인덱스 정보와 프롬프트 해시를 실행 설정으로 반환합니다.
    def describe(self):
        from rag_context import LLM_MODEL, CONTEXT_TOKEN_BUDGET, PER_PARENT_TOKEN_BUDGET
        from rag_reranking import RERANKER_MODEL
        return {'mode': self.mode, 'candidate_k': self.candidate_k, 'llm': LLM_MODEL,
                'reranker': RERANKER_MODEL, 'context_token_budget': CONTEXT_TOKEN_BUDGET,
                'per_parent_budget': PER_PARENT_TOKEN_BUDGET, 'index_manifest': self.manifest,
                'prompt_sha256': hashlib.sha256(self.prompt.encode()).hexdigest()}

    # 질문으로 child 후보를 찾은 뒤 리랭킹하여 상위 parent를 반환합니다.
    def retrieve(self, question, k):
        from collections import defaultdict
        from rag_reranking import rerank_to_parents
        # 후보 수보다 많은 parent를 요청하지 않도록 검색 깊이를 제한합니다.
        if not 1 <= k <= self.candidate_k:
            raise ValueError('K must be between 1 and candidate_k')
        # FAISS에서 의미 유사도가 높은 child 후보를 먼저 가져옵니다.
        children = self.store.similarity_search(question, k=self.candidate_k)
        # hybrid 실행에서는 의미 검색과 키워드 검색의 순위를 RRF로 합칩니다.
        if self.bm25:
            scores, documents = defaultdict(float), {}
            for ranking in (children, self.bm25.invoke(question)):
                for rank, doc in enumerate(ranking, 1):
                    # 같은 child를 두 검색 방식에서 찾았을 때 하나의 후보로 묶습니다.
                    key = (doc.metadata['source'], doc.metadata['parent_id'], doc.metadata['chunk_id'])
                    documents[key] = doc
                    # 점수 척도 대신 순위의 역수를 더하여 서로 다른 검색기를 결합합니다.
                    scores[key] += 1 / (60 + rank)
            children = [documents[key] for key in sorted(scores, key=scores.get, reverse=True)[:self.candidate_k]]
        # 통합 후보를 교차 인코더로 재정렬하고 같은 parent는 한 번만 반환합니다.
        parents = rerank_to_parents(question, children, self.parents, candidate_k=self.candidate_k, parent_k=k)
        result = []
        for doc in parents:
            # 컴퓨터마다 달라지는 절대 경로 대신 공통 상대 경로를 문서 ID로 사용합니다.
            corpus_id = doc.metadata.get('relative_path')
            if not corpus_id:
                raise ValueError('Index metadata requires relative_path for shared corpus IDs')
            # parent 원문과 리랭킹 점수 및 매칭 child 정보를 공통 문서 구조로 변환합니다.
            result.append(RetrievedDocument(corpus_id, doc.page_content,
                          f"{corpus_id}::{doc.metadata['parent_id']}",
                          doc.metadata.get('reranker_score'), dict(doc.metadata)))
        return result

    # 전달받은 검색 결과를 포맷한 뒤 질문과 함께 모델에 넣어 답변을 생성합니다.
    def generate(self, question, documents):
        from langchain_core.documents import Document
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from rag_context import format_context
        # 토큰 예산에 맞춰 매칭 child 주변을 발췌하고 출처 헤더를 붙입니다.
        context = format_context([Document(page_content=d.text, metadata=d.metadata) for d in documents])
        # 프롬프트와 생성 모델 및 문자열 파서를 연결하여 최종 답변을 얻습니다.
        chain = PromptTemplate.from_template(self.prompt) | self.llm | StrOutputParser()
        response = chain.invoke({'question': question, 'context': context})
        # 모델에 넣은 실제 맥락을 답변과 함께 반환하여 RAGAS 입력을 정확히 보존합니다.
        return Generation(response, [context] if context else [])


# CLI가 인자 없이 호출할 수 있도록 dense 어댑터 생성 함수를 제공합니다.
def dense():
    return OpenAIRAG(mode='dense')


# CLI가 인자 없이 호출할 수 있도록 hybrid 어댑터 생성 함수를 제공합니다.
def hybrid():
    return OpenAIRAG(mode='hybrid')

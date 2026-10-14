"""Portable RAG evaluation contracts. Optional providers are loaded lazily."""
# 공통 자료형과 인터페이스를 패키지 최상위에서 가져올 수 있도록 공개합니다.
from .contracts import RetrievedDocument, Generation, RAG

# 외부 구현자가 사용할 공개 이름을 명시하여 패키지 내부 구현과 구분합니다.
__all__ = ['RetrievedDocument', 'Generation', 'RAG']

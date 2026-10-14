import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from rag_eval import RetrievedDocument, Generation
from rag_eval.retrieval import ranking_metrics
from rag_eval.runner import collect, evaluate_retrieval
from rag_eval.ragas_scoring import score_saved


# 검색 계산과 저장 결과 재평가를 외부 API 없이 검증하는 테스트 모음입니다.
class EvaluationTests(unittest.TestCase):
    # 관련도 등급과 중복 검색 및 빈 검색에서 지표가 정의대로 계산되는지 확인합니다.
    def test_ranking(self):
        scores = ranking_metrics(['x', 'a', 'a', 'b'], {'a': 1, 'b': 2}, [1, 3, 4])
        self.assertEqual(scores['mrr'], .5)
        self.assertEqual(scores['recall@3'], .5)
        self.assertEqual(scores['recall@4'], 1)
        # 정답 등급과 순위로 nDCG를 직접 계산하여 구현 결과와 비교합니다.
        expected = (1 / math.log2(3) + 3 / math.log2(5)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(scores['ndcg@4'], expected)
        self.assertEqual(ranking_metrics([], {'a': 1}, [5])['ndcg@5'], 0)
        with self.assertRaises(ValueError):
            ranking_metrics(['a'], {}, [1])

    # 한 번 수집한 결과를 여러 번 평가해도 검색과 생성 호출이 늘지 않는지 확인합니다.
    def test_persistence_and_independent_scoring(self):
        # 실제 검색기와 생성기 대신 호출 횟수를 기록하는 가짜 RAG를 사용합니다.
        class FakeRAG:
            name = 'fake'
            retrievals = generations = 0
            def describe(self): return {'model': 'fake'}
            def retrieve(self, question, k):
                self.retrievals += 1
                return [RetrievedDocument('a', 'full parent', 'a::1')]
            def generate(self, question, docs):
                self.generations += 1
                return Generation('answer', ['actual excerpt'])
        # 실제 전달 맥락을 검사하고 고정 점수를 반환하는 가짜 채점기를 사용합니다.
        class Metric:
            async def ascore(self, **kwargs):
                if 'retrieved_contexts' in kwargs:
                    assert kwargs['retrieved_contexts'] == ['actual excerpt']
                return SimpleNamespace(value=.75)
        # 채점 실패를 의도적으로 발생시켜 실패 건수와 오류 저장 방식을 검증합니다.
        class Failure:
            async def ascore(self, **kwargs): raise RuntimeError('private payload')
        # 테스트가 만드는 수집 및 채점 파일을 임시 디렉터리에 격리합니다.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'samples.json'
            rag = FakeRAG()
            cases = [{'case_id': 'q1', 'user_input': 'question', 'reference': 'truth', 'relevance': {'a': 1}}]
            collect(rag, cases, path, k=3)
            # 파일을 다시 읽어 메모리 객체가 아닌 저장 결과로 재평가하는 경로를 검증합니다.
            bundle = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(evaluate_retrieval(bundle, [3])['summary']['recall@3'], 1)
            with self.assertRaises(ValueError): evaluate_retrieval(bundle, [5])
            with self.assertRaises(FileExistsError): collect(rag, cases, path)
            metrics = {'context_recall': Metric(), 'faithfulness': Failure(), 'factual_correctness': Metric()}
            # 같은 저장 데이터를 반복 채점하여 답변 재생성 없이 재사용되는지 검사합니다.
            for i in range(2):
                result = asyncio.run(score_saved(bundle, Path(directory) / f'scores{i}.json', metrics=metrics))
                self.assertEqual(result['summary']['faithfulness']['failure'], 1)
                self.assertIsNone(result['rows'][0]['faithfulness'])
                self.assertEqual(result['rows'][0]['faithfulness_error'], 'RuntimeError')
            # 재평가 횟수와 관계없이 검색과 생성이 각각 한 번만 호출되었는지 확인합니다.
            self.assertEqual((rag.retrievals, rag.generations), (1, 1))


if __name__ == '__main__':
    unittest.main()

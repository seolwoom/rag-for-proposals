"""Routing/metadata tests without model downloads or OpenAI requests."""
import ast
from collections import defaultdict
import json
from pathlib import Path
import unittest
from langchain_core.documents import Document
from rag_reranking import rerank_children, rerank_to_parents


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, tokens):
        return ''.join(tokens)

    def num_special_tokens_to_add(self, pair=True):
        return 4


class Scorer:
    tokenizer = Tokenizer()

    def __init__(self):
        self.pairs = []

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        return [10.0 if '정답' in passage else 1.0 for _, passage in pairs]


def child(i, parent, text):
    return Document(page_content=text, metadata={
        'source': 'pdf', 'parent_id': parent, 'chunk_id': i, 'pages': [i]})


class RerankingTests(unittest.TestCase):
    def test_ranking_parent_dedup_and_no_mutation(self):
        candidates = [child(0, 'a', '관련 없음'), child(1, 'b', '정답'), child(2, 'b', '정답')]
        parents = {key: Document(page_content=key, metadata={'parent_id': key, 'pages': [0, 1]})
                   for key in ['a', 'b']}
        result = rerank_to_parents('질문', candidates, parents, model=Scorer())
        self.assertEqual([d.metadata['parent_id'] for d in result], ['b', 'a'])
        self.assertEqual(result[0].metadata['matched_child_pages'], [1])
        self.assertEqual(result[0].metadata['reranker_score'], 10.0)
        self.assertNotIn('reranker_score', parents['b'].metadata)
        self.assertNotIn('reranker_score', candidates[1].metadata)

    def test_large_table_tail_is_scored(self):
        model = Scorer()
        table = child(0, 'a', '| 표 |\n' + '내용 ' * 900 + '정답')
        result = rerank_children('질문', [table], model=model)
        self.assertGreater(result[0].metadata['reranker_window_count'], 1)
        self.assertEqual(result[0].metadata['reranker_score'], 10.0)
        self.assertEqual(result[0].page_content, table.page_content)
        self.assertTrue(all(len(q) + len(p) + 4 <= 1024 for q, p in model.pairs))

    def test_cap_duplicate_candidates_and_empty_input(self):
        model = Scorer()
        docs = [child(i, str(i), '본문') for i in range(25)]
        self.assertEqual(len(rerank_children('질문', [docs[0]] + docs, model=model)), 20)
        self.assertEqual(len(model.pairs), 20)
        self.assertEqual(rerank_children('질문', []), [])

    def test_notebook_dense_saved_and_hybrid_routes(self):
        nb = json.loads(Path('중급 프로젝트_openai_다중문서_RAG.ipynb').read_text(encoding='utf-8'))
        for name in ['retriever', 'loaded_retriever']:
            source = next(''.join(c['source']) for c in nb['cells']
                          if c['cell_type'] == 'code' and f'\n{name} = RunnableLambda' in ''.join(c['source']))
            tree = ast.parse(source)
            assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
            called = []
            class Store:
                def similarity_search(self, query, k):
                    called.append(k)
                    return ['child']
            scope = {'RunnableLambda': lambda f: f, 'rerank_to_parents': lambda *a, **kw: (a, kw),
                     'vector_store': Store(), 'loaded_vector_store': Store(),
                     'parent_documents': {}, 'loaded_parent_documents': {}, 'CANDIDATE_K': 20, 'PARENT_K': 4}
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<route>', 'exec'), scope)
            args, kwargs = scope[name]('질문')
            self.assertEqual(called, [20])
            self.assertEqual(args[:2], ('질문', ['child']))
            self.assertEqual(kwargs['parent_k'], 4)
        source = next(''.join(c['source']) for c in nb['cells']
                      if c['cell_type'] == 'code' and 'def hybrid_search(query):' in ''.join(c['source']))
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        dense = [child(i, str(i), '본문') for i in range(20)]
        sparse = [child(i + 20, str(i + 20), '본문') for i in range(20)]
        class Dense:
            def similarity_search(self, query, k):
                return dense[:k]
        class Sparse:
            def invoke(self, query):
                return sparse
        scope = {'defaultdict': defaultdict, 'vector_store': Dense(), 'bm25_retriever': Sparse(),
                 'dense_k': 20, 'rrf_constant': 60, 'CANDIDATE_K': 20, 'final_k': 4,
                 'parent_documents': {}, 'rerank_to_parents': lambda *a, **kw: (a, kw)}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<hybrid>', 'exec'), scope)
        args, kwargs = scope['hybrid_search']('질문')
        self.assertEqual(len(args[1]), 20)
        self.assertEqual(kwargs['parent_k'], 4)


if __name__ == '__main__':
    unittest.main()

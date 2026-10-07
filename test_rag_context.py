import unittest
from langchain_core.documents import Document
from rag_chunking import build_parent_children
from rag_context import format_context, count_tokens


class ContextTests(unittest.TestCase):
    def test_table_heading_and_bounded_parent_offsets(self):
        docs = [Document(page_content='| Ⅰ. 추진개요 |\n| --- |\n내용\n\n'
                         '| 2 |  | 사업개요 |\n| --- | --- | --- |\n'
                         + '사업 일정과 예산 설명입니다.\n' * 3000,
                         metadata={'source': 'hwp'})]
        parents, children = build_parent_children(docs, 800, 150)
        self.assertIn('Ⅰ. 추진개요', [d.metadata['section_title'] for d in parents.values()])
        self.assertIn('2. 사업개요', [d.metadata['section_title'] for d in parents.values()])
        self.assertTrue(all(len(d.page_content) <= 8000 for d in parents.values()))
        for child in children:
            m = child.metadata
            self.assertEqual(parents[m['parent_id']].page_content[m['start_index']:m['end_index']],
                             child.page_content)

    def test_context_keeps_late_evidence_and_caps_total(self):
        text = '무관한 앞부분 설명입니다. ' * 10000 + '정답: 수행기간은 150일입니다.' + ' 뒤쪽 설명.' * 10000
        docs = [Document(page_content=text, metadata={
            'source': 'hwp', 'parent_id': str(i), 'matched_start_index': text.index('정답:')}) for i in range(4)]
        result = format_context(docs)
        self.assertLessEqual(count_tokens(result), 8000)
        self.assertEqual(result.count('정답: 수행기간은 150일입니다.'), 4)
        self.assertIn('[관련 구간 발췌]', result)

    def test_tiny_budget_and_empty_input(self):
        self.assertEqual(format_context([]), '')
        doc = Document(page_content='본문', metadata={'source': '긴 파일명' * 100})
        self.assertLessEqual(count_tokens(format_context([doc], token_budget=10)), 10)


if __name__ == '__main__':
    unittest.main()

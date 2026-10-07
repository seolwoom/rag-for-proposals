import json
import unittest
from langchain_core.documents import Document
from rag_chunking import build_parent_children


class ChildPageMetadataTests(unittest.TestCase):
    def check_offsets(self, parents, children):
        for child in children:
            m = child.metadata
            parent = parents[m['parent_id']]
            self.assertEqual(parent.page_content[m['start_index']:m['end_index']], child.page_content)
            self.assertEqual(m['parent_pages'], parent.metadata['pages'])
            self.assertEqual(json.loads(json.dumps(m)), m)

    def test_repeated_text_and_overlap(self):
        pages = [Document(page_content='   # Section\n' + 'Repeated content. ' * 30,
                          metadata={'source': 'pdf', 'page': i}) for i in range(2)]
        # Same heading would create separate sections; keep second page in first section.
        pages[1].page_content = 'Repeated content. ' * 30
        parents, children = build_parent_children(pages, 100, 30)
        self.assertEqual(len(parents), 1)
        self.assertEqual(next(iter(parents.values())).metadata['pages'], [0, 1])
        self.assertEqual(children[0].metadata['pages'], [0])
        self.assertEqual(children[-1].metadata['pages'], [1])
        self.check_offsets(parents, children)

    def test_prose_child_across_pages(self):
        parents, children = build_parent_children([
            Document(page_content='  # Section\nFirst page', metadata={'source': 'pdf', 'page': 0}),
            Document(page_content='Second page  ', metadata={'source': 'pdf', 'page': 1})])
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].metadata['pages'], [0, 1])
        self.check_offsets(parents, children)

    def test_table_pages_and_following_prose(self):
        pages = [Document(page_content='# Section\nFirst page prose\n\n| A | B |\n| --- | --- |\n| x | 1 |',
                          metadata={'source': 'pdf', 'page': 3}),
                 Document(page_content='| y | 2 |\n\nLast page prose',
                          metadata={'source': 'pdf', 'page': 4})]
        parents, children = build_parent_children(pages, 30, 5)
        table = next(c for c in children if c.metadata['is_table'])
        self.assertEqual(table.metadata['pages'], [3, 4])
        self.assertEqual(table.metadata['page'], 3)
        self.assertEqual(table.metadata['page_end'], 4)
        self.assertEqual(children[-1].metadata['pages'], [4])
        self.assertEqual(children[0].metadata['pages'], [3])
        self.check_offsets(parents, children)

    def test_no_invented_hwp_pages(self):
        parents, children = build_parent_children([
            Document(page_content='# Section\nHWP content', metadata={'source': 'hwp'})])
        self.assertEqual(children[0].metadata['pages'], [])
        self.assertNotIn('page', children[0].metadata)
        self.check_offsets(parents, children)


if __name__ == '__main__':
    unittest.main()

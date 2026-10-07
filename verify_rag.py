"""Offline PDF extraction audit and parent-child invariants (no OpenAI calls)."""
import json
from pathlib import Path
import pymupdf4llm
from langchain_core.documents import Document
from rag_chunking import build_parent_children, expand_parents, markdown_table_count

def main():
    pages, report = [], []
    for path in sorted(Path('files').rglob('*')):
        if path.suffix.lower() != '.pdf':
            continue
        rows = pymupdf4llm.to_markdown(str(path), page_chunks=True)
        target = Path('extracted_markdown') / path.relative_to('files').with_suffix('.pdf.md')
        target.parent.mkdir(parents=True, exist_ok=True)
        text = '\n\n'.join(row['text'] for row in rows)
        target.write_text(text, encoding='utf-8')
        report.append({'source': str(path), 'markdown_path': str(target),
                       'page_count': len(rows), 'markdown_table_count': markdown_table_count(text)})
        pages.extend(Document(page_content=row['text'], metadata={'source': str(path), 'page': i})
                     for i, row in enumerate(rows) if row['text'].strip())
    parents, children = build_parent_children(pages)
    assert all(c.metadata['parent_id'] in parents for c in children)
    assert len({c.metadata['chunk_id'] for c in children}) == len(children)
    if children:
        assert len(expand_parents([children[0], children[0]], parents)) == 1
    table = '| 항목 | 금액 |\n| --- | --- |\n' + '| 사업 | 100 |\n' * 100
    fixture = [Document(page_content='# 사업\n내용\n' + table, metadata={'source': 'fixture', 'page': 0}),
               Document(page_content='계속되는 내용\n## 예산\n예산 설명', metadata={'source': 'fixture', 'page': 1})]
    fixture_parents, fixture_children = build_parent_children(fixture)
    assert len(fixture_parents) == 2
    assert next(iter(fixture_parents.values())).metadata['pages'] == [0, 1]
    assert any(c.page_content == table.strip() for c in fixture_children)
    Path('pdf_markdown_audit.json').write_text(json.dumps({
        'files': report, 'parent_count': len(parents), 'child_count': len(children),
        'note': 'Counts describe detected Markdown tables, not completeness against the original PDF.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f'PASS: {len(parents)} parents, {len(children)} children; table and cross-page invariants')

if __name__ == '__main__':
    main()

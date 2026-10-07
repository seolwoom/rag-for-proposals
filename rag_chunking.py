"""Section parents, table-preserving children, and parent expansion for RAG."""
import re
from collections import defaultdict

HEADING = re.compile(r'^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$')
NUMBERED = re.compile(r'^\s*((?:\d+\.)+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+[.、]?|[가-힣][.)])\s+(.{1,80})$')
TABLE_SEPARATOR = re.compile(r'^\s*\|?\s*:?-{3,}:?\s*\|.*$')


def markdown_table_count(text):
    return sum(bool(TABLE_SEPARATOR.match(line)) for line in text.splitlines())


def build_parent_children(pages, child_size=500, child_overlap=100, parent_size=8000):
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    if not 0 <= child_overlap < child_size:
        raise ValueError('Require 0 <= child_overlap < child_size')
    splitter = RecursiveCharacterTextSplitter(chunk_size=child_size,
        chunk_overlap=child_overlap, add_start_index=True)
    if parent_size < child_size:
        raise ValueError('parent_size must be >= child_size')
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size,
        chunk_overlap=0, add_start_index=True)
    grouped = defaultdict(list)
    for doc in pages:
        grouped[doc.metadata['source']].append(doc)
    parents, children = {}, []
    for source, source_pages in grouped.items():
        hierarchy, section_lines, line_pages = [], [], []
        base = dict(source_pages[0].metadata)

        def flush():
            raw_content = '\n'.join(section_lines)
            trim_start = len(raw_content) - len(raw_content.lstrip())
            content = raw_content.strip()
            if not content:
                return
            spans, offset = [], 0
            for line, page in zip(section_lines, line_pages):
                # Newline separators have no source page of their own.
                start = max(0, offset - trim_start)
                end = min(len(content), offset + len(line) - trim_start)
                if page is not None and end > start and line.strip():
                    spans.append((start, end, page))
                offset += len(line) + 1
            for part, segment in enumerate(parent_splitter.create_documents([content])):
                a = segment.metadata['start_index']
                b = a + len(segment.page_content)
                local_spans = [(max(0, x - a), min(b, y) - a, page)
                               for x, y, page in spans if x < b and y > a]
                emit_parent(segment.page_content, local_spans, part)

        def emit_parent(content, spans, part):
            parent_id = f'parent-{len(parents)}'
            section_pages = list(dict.fromkeys(page for _, _, page in spans))
            metadata = {k: v for k, v in base.items()
                        if k not in ('page', 'page_label', 'markdown_table_count')}
            metadata.update(parent_id=parent_id, section_part=part, section_title=hierarchy[-1][1] if hierarchy else '본문',
                section_path=[title for _, title in hierarchy], pages=list(dict.fromkeys(section_pages)))
            if section_pages:
                metadata.update(page=min(section_pages), page_end=max(section_pages))
            parents[parent_id] = Document(page_content=content, metadata=metadata)
            # Tables remain atomic within each bounded parent part.
            lines = content.splitlines(keepends=True)
            blocks, prose, i, offset, prose_start = [], [], 0, 0, 0

            def split_prose():
                for chunk in splitter.create_documents([''.join(prose)]):
                    start = prose_start + chunk.metadata['start_index']
                    blocks.append((chunk.page_content, start, start + len(chunk.page_content)))

            while i < len(lines):
                if i + 1 < len(lines) and TABLE_SEPARATOR.match(lines[i + 1]):
                    if prose:
                        split_prose()
                        prose = []
                    table_start = offset
                    table = [lines[i], lines[i + 1]]
                    offset += len(lines[i]) + len(lines[i + 1])
                    i += 2
                    while i < len(lines) and '|' in lines[i] and lines[i].strip():
                        table.append(lines[i])
                        offset += len(lines[i])
                        i += 1
                    raw_table = ''.join(table)
                    text = raw_table.strip()
                    start = table_start + len(raw_table) - len(raw_table.lstrip())
                    blocks.append((text, start, start + len(text)))
                else:
                    if not prose:
                        prose_start = offset
                    prose.append(lines[i])
                    offset += len(lines[i])
                    i += 1
            if prose:
                split_prose()
            for block, start, end in blocks:
                if block.strip():
                    child_pages = list(dict.fromkeys(page for a, b, page in spans
                                                    if a < end and b > start))
                    child_metadata = {**metadata, 'parent_pages': section_pages,
                        'pages': child_pages, 'start_index': start, 'end_index': end,
                        'chunk_id': len(children), 'is_table': markdown_table_count(block) > 0}
                    child_metadata.pop('page', None)
                    child_metadata.pop('page_end', None)
                    if child_pages:
                        child_metadata.update(page=min(child_pages), page_end=max(child_pages))
                    children.append(Document(page_content=block, metadata=child_metadata))

        for doc in source_pages:
            for line in doc.page_content.splitlines():
                heading = HEADING.match(line)
                title_line = line
                if line.strip().startswith('|'):
                    cells = [cell.strip() for cell in line.strip().strip('|').split('|') if cell.strip()]
                    if len(cells) == 1 and len(cells[0]) <= 80:
                        title_line = cells[0]
                    elif (len(cells) == 2 and re.fullmatch(r'\d{1,2}', cells[0])
                          and len(cells[1]) <= 40):
                        title_line = cells[0] + '. ' + cells[1]
                numbered = NUMBERED.match(title_line) if not heading and '|' not in title_line else None
                if heading or numbered:
                    flush()
                    section_lines, line_pages = [], []
                    level = len(heading[1]) if heading else 2
                    title = heading[2] if heading else title_line.strip()
                    while hierarchy and hierarchy[-1][0] >= level:
                        hierarchy.pop()
                    hierarchy.append((level, title))
                section_lines.append(line)
                line_pages.append(doc.metadata.get('page'))
        flush()
    return parents, children


def expand_parents(children, parents, limit=4):
    result, seen = [], set()
    for child in children:
        parent_id = child.metadata['parent_id']
        if parent_id not in seen:
            result.append(parents[parent_id])
            seen.add(parent_id)
            if len(result) >= limit:
                break
    return result

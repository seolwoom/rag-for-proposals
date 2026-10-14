"""Index building extracted from notebook v0.3; run explicitly, never on import."""

# 노트북의 문서 로딩과 청킹 및 인덱스 저장을 명시적 함수 호출로 옮겼습니다.
def build_index(files_dir="./files", index_dir="./faiss_index"):
    """hwp-hwpx-parser 1.0.0의 잘린 필드 제어문자 무한 루프 수정.
    본문/표/각주 추출 로직은 유지합니다. 원본 문서를 변경하지 않습니다.
    """
    import struct
    from typing import List
    from hwp_hwpx_parser.hwp5 import HWP5Reader, CTRL_ID_HYPERLINK
    
    # 손상된 HWP 필드에서도 읽기 위치를 전진시키도록 하이퍼링크 추출을 보정합니다.
    def _extract_hyperlink_texts_from_para(self, para_data: bytes) -> List[str]:
        hyperlink_texts = []
        i = 0
    
        # UTF-16 제어 코드를 읽을 최소 바이트가 남아 있는 동안 순회합니다.
        while i < len(para_data) - 1:
            code = struct.unpack_from("<H", para_data, i)[0]
    
            # 필드 시작 제어 문자를 만나면 제어 ID와 텍스트 범위를 확인합니다.
            if code == 0x03:
                if i + 6 > len(para_data):
                    i += 2
                    continue
                if i + 6 <= len(para_data):
                    ctrl_id = struct.unpack_from("<I", para_data, i + 2)[0]
    
                    if ctrl_id == CTRL_ID_HYPERLINK:
                        text_start = i + 14
                        text_chars = []
                        j = text_start
    
                        while j < len(para_data) - 1:
                            c = struct.unpack_from("<H", para_data, j)[0]
                            if c == 0x04:
                                break
                            elif c == 0x03:
                                j += 2
                            elif 0x20 <= c < 0x10000:
                                text_chars.append(chr(c))
                                j += 2
                            else:
                                j += 2
    
                        if text_chars:
                            hyperlink_texts.append("".join(text_chars))
                        i = j
                        continue
                    else:
                        i += 14
                        continue
            elif code == 0x04:
                i += 10
            elif code < 32:
                if code in (11, 12):
                    i += 10
                elif 15 <= code <= 23:
                    i += 14
                else:
                    i += 2
            else:
                i += 2
    
        return hyperlink_texts
    
    
    # 현재 프로세스의 파서 메서드만 교체하며 원본 문서 파일을 수정하지 않습니다.
    def apply_hwp_parser_fix():
        HWP5Reader._extract_hyperlink_texts_from_para = _extract_hyperlink_texts_from_para
    
    # 문서를 읽기 전에 파서 보정을 적용하여 손상된 필드의 반복을 방지합니다.
    apply_hwp_parser_fix()
    print("HWP 파서의 제어문자 반복문 수정 적용 완료")
    
    from pathlib import Path
    import json
    import pymupdf4llm
    from langchain_core.documents import Document
    from rag_chunking import markdown_table_count
    from langchain_hwp_hwpx import HwpHwpxLoader
    from hwp_hwpx_parser import ExtractOptions, TableStyle
    
    # 호출자가 지정한 문서 폴더를 절대 경로로 변환하여 출처 정보를 일관되게 기록합니다.
    FILES_DIR = Path(files_dir).resolve()
    # 인덱스와 청크 및 manifest를 함께 저장할 디렉터리를 지정합니다.
    INDEX_DIR = Path(index_dir).resolve()
    EXPECTED_FILE_COUNT = 100  # 확인용이며 파일 수를 제한하지 않습니다.
    STOP_ON_LOAD_ERROR = True
    
    print("현재 작업 폴더:", Path.cwd())
    print("문서 폴더:", FILES_DIR)
    if not FILES_DIR.is_dir():
        raise FileNotFoundError(f"문서 폴더가 없습니다: {FILES_DIR}")
    # 문서 폴더 아래의 PDF와 HWP를 재귀적으로 찾고 일정한 순서로 정렬합니다.
    file_paths = sorted(
        (p for p in FILES_DIR.rglob("*")
         if p.is_file() and p.suffix.lower() in {".pdf", ".hwp"}),
        key=lambda p: str(p).lower(),
    )
    if not file_paths:
        raise ValueError("./files에서 PDF 또는 HWP 파일을 찾지 못했습니다.")
    print(f"발견한 파일: {len(file_paths)}개")
    if len(file_paths) != EXPECTED_FILE_COUNT:
        print(f"참고: 예상 {EXPECTED_FILE_COUNT}개와 다릅니다. 발견한 파일 전체를 처리합니다.")
    
    # 로더가 추출한 텍스트를 모아 parent와 child 생성의 입력으로 사용합니다.
    pages = []  # PDF 페이지와 HWP 문서를 담는 기존 RAG 입력 변수
    loaded_files = []
    load_errors = []
    empty_pdf_pages = []
    pdf_markdown_report = []
    MARKDOWN_DIR = Path("./extracted_markdown").resolve()
    # 파일별로 추출을 시도하고 성공 내역과 실패 내역을 따로 기록합니다.
    for number, file_path in enumerate(file_paths, start=1):
        try:
            # PDF는 페이지별 Markdown으로 추출하여 페이지와 표 관련 정보를 보존합니다.
            if file_path.suffix.lower() == ".pdf":
                pdf_pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
                documents = []
                markdown_path = MARKDOWN_DIR / file_path.relative_to(FILES_DIR).with_suffix('.pdf.md')
                markdown_path.parent.mkdir(parents=True, exist_ok=True)
                markdown_text = '\n\n'.join(row['text'] for row in pdf_pages)
                markdown_path.write_text(markdown_text, encoding='utf-8')
                for page_number, row in enumerate(pdf_pages):
                    documents.append(Document(page_content=row['text'], metadata={
                        'page': page_number, 'markdown_path': str(markdown_path),
                        'markdown_table_count': markdown_table_count(row['text'])}))
                pdf_markdown_report.append({'source': str(file_path),
                    'markdown_path': str(markdown_path), 'page_count': len(pdf_pages),
                    'markdown_table_count': markdown_table_count(markdown_text)})
            else:
                # HWP는 본문과 표 및 각주를 포함해 하나의 문서로 읽습니다.
                loader = HwpHwpxLoader(
                    file_path=file_path,
                    mode="single",
                    extract_options=ExtractOptions(table_style=TableStyle.MARKDOWN),
                    include_tables=True,
                    include_notes=True,
                    on_encrypted="raise",
                    on_invalid="raise",
                    on_error="raise",
                )
                documents = loader.load()
            valid_documents = []
            # 빈 텍스트는 제외하고 공통 출처 메타데이터를 각 추출 문서에 추가합니다.
            for doc in documents:
                if not doc.page_content.strip():
                    if file_path.suffix.lower() == ".pdf":
                        empty_pdf_pages.append({
                            "source": str(file_path), "page": doc.metadata.get("page")
                        })
                    continue
                # 상대 경로와 파일 종류를 보존하여 이후 검색 결과를 공통 원문 ID에 연결합니다.
                doc.metadata.update({
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "relative_path": file_path.relative_to(FILES_DIR).as_posix(),
                    "file_type": file_path.suffix.lower().lstrip("."),
                })
                valid_documents.append(doc)
            if not valid_documents:
                raise ValueError("추출된 텍스트가 없습니다. 스캔/OCR 또는 문서 형식을 확인하세요.")
            pages.extend(valid_documents)
            loaded_files.append(str(file_path))
            print(f"[{number}/{len(file_paths)}] 성공: {file_path.name} ({len(valid_documents)}개 문서 단위)")
        # 개별 파일 실패를 보고서에 모아 누락 문서를 확인할 수 있게 합니다.
        except Exception as exc:
            load_errors.append({"source": str(file_path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{number}/{len(file_paths)}] 실패: {file_path.name} - {exc}")
    
    print(f"파일 로딩: 성공 {len(loaded_files)}개 / 실패 {len(load_errors)}개")
    print(f"텍스트가 없는 PDF 페이지: {len(empty_pdf_pages)}개 (OCR 필요 여부 확인)")
    # 추출 성공과 오류 및 빈 페이지 내역을 별도 보고서로 기록합니다.
    Path("./document_load_report.json").write_text(
        json.dumps({"loaded_files": loaded_files, "load_errors": load_errors,
                    "empty_pdf_pages": empty_pdf_pages,
                    "pdf_markdown_report": pdf_markdown_report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 기본 설정에서는 문서 로딩 오류가 있으면 불완전한 코퍼스로 인덱스를 만들지 않습니다.
    if load_errors and STOP_ON_LOAD_ERROR:
        raise RuntimeError("읽기 실패 파일이 있습니다. document_load_report.json을 확인하고 재실행하세요.")
    if not pages:
        raise ValueError("청킹할 문서 텍스트가 없습니다.")
    
    
    from rag_chunking import build_parent_children, expand_parents
    from langchain_core.runnables import RunnableLambda
    
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    PARENT_SIZE = 8000
    # 섹션 중심 parent와 검색용 child를 기존 청킹 모듈로 생성합니다.
    parent_documents, all_splits = build_parent_children(pages, CHUNK_SIZE, CHUNK_OVERLAP, parent_size=PARENT_SIZE)
    if not all_splits:
        raise ValueError("생성된 child 청크가 없습니다.")
    print("섹션 parent 수:", len(parent_documents))
    print("검색 child 수:", len(all_splits))
    
    
    from datetime import datetime, timezone
    from rag_reranking import rerank_to_parents, RERANKER_MODEL, CANDIDATE_K, PARENT_K
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    
    EMBEDDING_MODEL = "text-embedding-3-small"
    BATCH_SIZE = 64
    # 문서 벡터와 이후 질의 벡터에 사용할 OpenAI 임베딩 모델을 준비합니다.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=BATCH_SIZE)
    
    # 첫 배치의 실제 임베딩 차원에 맞춰 FAISS 인덱스를 생성합니다.
    # 첫 child 배치로 FAISS를 만들고 나머지 child를 같은 순서로 추가합니다.
    vector_store = None
    for start in range(0, len(all_splits), BATCH_SIZE):
        batch = all_splits[start:start + BATCH_SIZE]
        ids = [str(doc.metadata["chunk_id"]) for doc in batch]
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings, ids=ids)
        else:
            vector_store.add_documents(batch, ids=ids)
        print(f"임베딩 및 FAISS 저장: {min(start + BATCH_SIZE, len(all_splits))}/{len(all_splits)} 청크")
    
    # index.faiss: 벡터 인덱스, index.pkl: 문서/메타데이터와 ID 매핑
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # 벡터 검색 인덱스를 저장하여 후속 실행에서 문서 임베딩을 재사용합니다.
    vector_store.save_local(str(INDEX_DIR))
    # BM25 재구축에도 사용할 수 있도록 청크를 JSON으로 저장합니다.
    # child 텍스트와 메타데이터를 JSON으로 저장하여 BM25와 docstore를 복원합니다.
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps([{"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in all_splits], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 생성 맥락으로 확장할 parent 원문을 ID별로 저장합니다.
    (INDEX_DIR / "parents.json").write_text(
        json.dumps({key: {"page_content": doc.page_content, "metadata": doc.metadata}
                    for key, doc in parent_documents.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 청킹 설정과 임베딩 차원 및 파일 로딩 내역을 인덱스 설명 정보로 기록합니다.
    manifest = {
        "chunking_strategy": "section_parent_child_v2",
        "parent_size": PARENT_SIZE,
        "parent_count": len(parent_documents),
        "pdf_markdown_report": pdf_markdown_report,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files_dir": str(FILES_DIR),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": vector_store.index.d,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "file_count_found": len(file_paths),
        "file_count_loaded": len(loaded_files),
        "chunk_count": len(all_splits),
        "loaded_files": loaded_files,
        "load_errors": load_errors,
        "empty_pdf_pages": empty_pdf_pages,
    }
    # manifest를 함께 저장하여 다른 설정의 인덱스를 잘못 읽는 것을 방지합니다.
    (INDEX_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 노트북에서 사용하던 검색기를 구성하며 실제 질의 실행은 이 함수에서 수행하지 않습니다.
    retriever = RunnableLambda(lambda query: rerank_to_parents(
        query, vector_store.similarity_search(query, k=CANDIDATE_K),
        parent_documents, candidate_k=CANDIDATE_K, parent_k=PARENT_K))
    print("FAISS 로컬 저장 완료:", INDEX_DIR)
    print("벡터 개수:", vector_store.index.ntotal)

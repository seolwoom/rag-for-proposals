"""Python entrypoint replacing execution of notebook v0.3 cells.

See EVALUATION.md. Importing this module does not load models or call APIs.
"""
# 노트북을 실행하는 대신 모듈화된 명령줄 실행기를 재사용합니다.
from rag_eval.__main__ import main

# 직접 실행할 때만 명령을 처리하고 import 시에는 모델이나 API를 실행하지 않습니다.
if __name__ == '__main__':
    main()

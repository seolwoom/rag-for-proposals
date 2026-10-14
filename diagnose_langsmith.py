"""Read-only LangSmith connectivity check; never prints keys or run content."""
import logging
import os
from itertools import islice
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import tracing_is_enabled


def main():
    logging.getLogger("langsmith").setLevel(logging.CRITICAL)
    load_dotenv(Path(__file__).with_name(".env"), override=True, encoding="utf-8-sig")
    print("tracing_enabled:", tracing_is_enabled())
    project_name = os.getenv("LANGSMITH_PROJECT", "rag-for-proposals")
    print("project:", project_name)
    try:
        client = Client(timeout_ms=10000)
        projects = list(islice(client.list_projects(name=project_name), 1))
        print("project_matches:", len(projects))
        for project in projects:
            runs = list(islice(client.list_runs(project_id=project.id), 5))
            print("recent_run_count:", len(runs))
            for run in runs:
                print("run:", run.name, "started:", run.start_time, "status:", run.status)
    except Exception as exc:
        print("connection_error_type:", type(exc).__name__)
        if isinstance(exc, TypeError):
            print("argument_error:", str(exc))


if __name__ == "__main__":
    main()

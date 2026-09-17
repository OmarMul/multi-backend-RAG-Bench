"""Tracing and observability wrapper for RAG pipelines (LangSmith + Langfuse)."""

import functools
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

from src.config import settings

logger = logging.getLogger("rag_bench.tracing")

# 1. Setup LangSmith environment
if settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

# 2. Setup Langfuse environment & client
langfuse_client = None
if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
    host_url = settings.LANGFUSE_BASE_URL or settings.LANGFUSE_HOST or "https://cloud.langfuse.com"
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = host_url
    try:
        from langfuse import Langfuse
        langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=host_url,
        )
    except Exception as e:
        logger.warning(f"Could not initialize Langfuse client: {e}")


def traced(span_name: Optional[str] = None):
    """Decorator to log latency, tokens, parameters, and metadata of RAG operations to LangSmith and Langfuse."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not settings.ENABLE_TRACING:
                return func(*args, **kwargs)

            name = span_name or func.__name__
            start_time = time.perf_counter()

            # Extract pipeline instance if called as a method
            instance = args[0] if args else None
            framework = getattr(instance, "framework_name", "rag_pipeline")
            backend = getattr(instance, "backend_name", "unknown")
            model = getattr(instance, "model_name", "unknown")

            question = kwargs.get("question") or (args[1] if len(args) > 1 else "N/A")
            top_k = kwargs.get("top_k", 4)
            full_span_name = f"{framework}_{name}"

            # Execute with LangSmith traceable run if enabled
            def _execute():
                return func(*args, **kwargs)

            # Check if LangSmith traceable is available
            if settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
                try:
                    from langsmith import traceable
                    _execute_traced = traceable(
                        name=full_span_name,
                        project_name=settings.LANGCHAIN_PROJECT,
                        run_type="chain",
                        metadata={
                            "framework": framework,
                            "vector_backend": backend,
                            "model": model,
                            "top_k": top_k,
                        }
                    )(_execute)
                except Exception:
                    _execute_traced = _execute
            else:
                _execute_traced = _execute

            # Check if Langfuse observe is available
            if langfuse_client:
                try:
                    from langfuse import observe
                    _execute_observed = observe(name=full_span_name)(_execute_traced)
                except Exception:
                    _execute_observed = _execute_traced
            else:
                _execute_observed = _execute_traced

            try:
                result = _execute_observed()
                elapsed = time.perf_counter() - start_time

                chunk_count = len(getattr(result, "sources", []))
                print(
                    f"\n[TRACE] {framework.upper()} ({backend}) | "
                    f"Model: {model} | Latency: {elapsed:.3f}s | Chunks: {chunk_count}"
                )

                # Flush Langfuse buffer
                if langfuse_client:
                    try:
                        langfuse_client.flush()
                    except Exception:
                        pass

                return result

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                print(f"[TRACE ERROR] {framework.upper()} failed after {elapsed:.3f}s: {e}")
                if langfuse_client:
                    try:
                        langfuse_client.flush()
                    except Exception:
                        pass
                raise e

        return wrapper
    return decorator


if __name__ == "__main__":
    @traced("sample_test_run")
    def sample_func(question: str):
        time.sleep(0.05)
        return type("MockResponse", (), {"answer": "Sample answer", "sources": [1, 2, 3]})()

    print("Testing tracing decorator:")
    res = sample_func(question="What is Agentic RAG?")
    print("Test finished successfully.")
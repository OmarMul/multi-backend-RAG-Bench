"""API routes: POST /query, GET /runs, GET /runs/{id}/score."""

import time
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.config import settings, FrameworkType, VectorBackendType

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    framework: Optional[FrameworkType] = None
    vector_backend: Optional[VectorBackendType] = None
    model: Optional[str] = None
    top_k: int = 4
    run_eval: bool = False  # if True, also run RAGAS scoring
    ground_truth: Optional[str] = None  # needed for precision/recall


class QueryResponse(BaseModel):
    run_id: Optional[int]
    question: str
    answer: str
    framework: str
    vector_backend: str
    model: str
    latency_sec: float
    sources: list[str]
    scores: Optional[dict] = None


class RunSummary(BaseModel):
    id: int
    query: str
    framework: str
    vector_backend: str
    model: str
    latency_sec: float
    created_at: str


class ScoreResponse(BaseModel):
    run_id: int
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pipeline(framework: FrameworkType, vector_backend: VectorBackendType, model: str):
    """Instantiate the correct pipeline based on config."""
    if framework == FrameworkType.LANGCHAIN:
        from src.rag.langchain_pipeline import LangChainPipeline
        return LangChainPipeline(vector_backend=vector_backend, model=model)
    else:
        from src.rag.llamaindex_pipeline import LlamaIndexPipeline
        return LlamaIndexPipeline(vector_backend=vector_backend, model=model)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(req: QueryRequest):
    """Run a question through a RAG pipeline and return the answer + sources."""
    framework = req.framework or settings.RAG_FRAMEWORK
    vector_backend = req.vector_backend or settings.VECTOR_BACKEND
    model = req.model or settings.LLM_MODEL

    try:
        pipeline = _get_pipeline(framework, vector_backend, model)
        response = pipeline.query(req.question, top_k=req.top_k)
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    # Optionally run RAGAS scoring
    scores = None
    if req.run_eval:
        try:
            from src.eval.ragas_eval import score_answer
            scores = score_answer(
                question=req.question,
                answer=response.answer,
                contexts=[getattr(c, "content", getattr(c, "text", str(c))) for c in response.sources],
                ground_truth=req.ground_truth,
            )
        except Exception as exc:
            logger.warning("RAGAS eval failed: %s", exc)

    # Resolve run_id from DB if available
    run_id: Optional[int] = None
    try:
        from src.db.session import get_session
        from src.db.models import Run
        with get_session() as session:
            run = (
                session.query(Run)
                .filter(Run.query == req.question)
                .order_by(Run.id.desc())
                .first()
            )
            if run:
                run_id = run.id
                # Attach scores if eval was run
                if scores and run_id:
                    from src.rag.base import BaseRAGPipeline
                    BaseRAGPipeline.save_scores(run_id, scores)
    except Exception:
        pass

    return QueryResponse(
        run_id=run_id,
        question=response.query,
        answer=response.answer,
        framework=response.framework or str(framework),
        vector_backend=response.vector_backend or str(vector_backend),
        model=response.model or model,
        latency_sec=response.latency_sec,
        sources=[getattr(c, "content", getattr(c, "text", str(c))) for c in response.sources],
        scores=scores,
    )


@router.get("/runs", response_model=list[RunSummary], tags=["runs"])
def list_runs(
    limit: int = Query(50, le=200),
    framework: Optional[FrameworkType] = None,
    vector_backend: Optional[VectorBackendType] = None,
):
    """List stored RAG query runs, optionally filtered."""
    try:
        from src.db.session import get_session, create_tables
        from src.db.models import Run
        create_tables()
        with get_session() as session:
            q = session.query(Run)
            if framework:
                q = q.filter(Run.framework == str(framework))
            if vector_backend:
                q = q.filter(Run.vector_backend == str(vector_backend))
            runs = q.order_by(Run.id.desc()).limit(limit).all()
            return [
                RunSummary(
                    id=r.id,
                    query=r.query,
                    framework=r.framework,
                    vector_backend=r.vector_backend,
                    model=r.model,
                    latency_sec=r.latency_sec,
                    created_at=r.created_at.isoformat(),
                )
                for r in runs
            ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/runs/{run_id}/score", response_model=ScoreResponse, tags=["runs"])
def get_score(run_id: int):
    """Fetch RAGAS scores for a specific run."""
    try:
        from src.db.session import get_session
        from src.db.models import Score
        with get_session() as session:
            score = session.query(Score).filter(Score.run_id == run_id).first()
            if not score:
                raise HTTPException(status_code=404, detail=f"No scores found for run_id={run_id}")
            return ScoreResponse(
                run_id=score.run_id,
                faithfulness=score.faithfulness,
                answer_relevancy=score.answer_relevancy,
                context_precision=score.context_precision,
                context_recall=score.context_recall,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/runs/{run_id}/eval", response_model=ScoreResponse, tags=["runs"])
def eval_run(run_id: int, ground_truth: Optional[str] = None):
    """Run RAGAS evaluation on an existing stored run."""
    try:
        from src.db.session import get_session
        from src.db.models import Run
        from src.eval.ragas_eval import score_answer
        from src.rag.base import BaseRAGPipeline

        with get_session() as session:
            run = session.query(Run).filter(Run.id == run_id).first()
            if not run:
                raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
            question, answer = run.query, run.answer

        scores = score_answer(question=question, answer=answer, contexts=[], ground_truth=ground_truth)
        BaseRAGPipeline.save_scores(run_id, scores)

        return ScoreResponse(
            run_id=run_id,
            faithfulness=scores.get("faithfulness"),
            answer_relevancy=scores.get("answer_relevancy"),
            context_precision=scores.get("context_precision"),
            context_recall=scores.get("context_recall"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

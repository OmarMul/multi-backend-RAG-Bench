"""RAGAS automated scoring evaluation for RAG-Bench pipelines."""


import json
from pathlib import Path
from typing import Dict, List, Optional, Any


from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig


from src.config import settings
from src.rag.base import BaseRAGPipeline, RAGResponse



class RagasEvaluator:
    """Evaluates RAG pipeline answers against standard RAGAS metrics using OpenRouter."""
    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.model_name = model_name or settings.LLM_MODEL
        self.api_key = settings.OPENROUTER_API_KEY


        # Initialize LLM judge via OpenRouter
        self.evaluator_llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
        )

        # Initialize HuggingFace embeddings for relevancy scoring
        self.evaluator_embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )



        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]


    def score_single(
        self,
        question: str,
        answer: str,
        contexts: str,
        ground_truth: str,
    ) -> Dict[str,float]:
        """Score a single Q&A generation on a 0.0 to 1.0 scale."""

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)


        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
            run_config=RunConfig(max_workers=2, max_wait=120),
        )
        return {
            "faithfulness": round(float(sum(results["faithfulness"]) / len(results["faithfulness"])), 4),
            "answer_relevancy": round(float(sum(results["answer_relevancy"]) / len(results["answer_relevancy"])), 4),
            "context_precision": round(float(sum(results["context_precision"]) / len(results["context_precision"])), 4),
            "context_recall": round(float(sum(results["context_recall"]) / len(results["context_recall"])), 4),
        }
    

    def evaluate_pipeline(
        self,
        pipeline: BaseRAGPipeline,
        eval_quetions_path: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Run the evaluation benchmark dataset against any RAG pipeline."""
        
        eval_file = Path(eval_quetions_path or "./data/raw_docs/eval_questions.json")


        if not eval_file.exists():
             raise FileNotFoundError(f"Evaluation questions file not found: {eval_file}")

        with open(eval_file, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)


        questions: List[str] = []
        answers: List[str] = []
        contexts: List[List[str]] = []
        ground_truths: List[str] = []
        latencies: List[float] = []

        print(f"\n🚀 Running RAGAS Evaluation on [{pipeline.framework_name.upper()} + {pipeline.backend_name.upper()}]...")

        for idx, item in enumerate(qa_pairs):
            q = item["question"]
            gt = item["ground_truth"]
            print(f"  [{idx + 1}/{len(qa_pairs)}] Querying: '{q[:50]}...'")

            res: RAGResponse = pipeline.query(question=q, top_k=top_k)

            questions.append(q)
            answers.append(res.answer)
            contexts.append([c.content for c in res.sources])
            ground_truths.append(gt)
            latencies.append(res.latency_sec)

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })


        print("Calculating Ragas Metrics (Faithfulness, Relevancy, Precision, Recall)...")
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
            run_config=RunConfig(max_workers=2, max_wait=120),
        )

        avg_latency = round(sum(latencies) / len(latencies), 4) if latencies else 0.0

        scores = {
            "framework": pipeline.framework_name,
            "vector_backend": pipeline.backend_name,
            "model": pipeline.model_name,
            "avg_latency_sec": avg_latency,
            "faithfulness": round(float(sum(results["faithfulness"]) / len(results["faithfulness"])), 4),
            "answer_relevancy": round(float(sum(results["answer_relevancy"]) / len(results["answer_relevancy"])), 4),
            "context_precision": round(float(sum(results["context_precision"]) / len(results["context_precision"])), 4),
            "context_recall": round(float(sum(results["context_recall"]) / len(results["context_recall"])), 4),
        }

        return scores

if __name__ == "__main__":
    from src.rag.langchain_pipeline import LangChainPipeline
    from src.config import VectorBackendType
    evaluator = RagasEvaluator()
    # Run benchmark evaluation on LangChain + Chroma
    pipeline = LangChainPipeline(backend=VectorBackendType.CHROMA)
    summary = evaluator.evaluate_pipeline(pipeline)
    print("\n" + "=" * 60)
    print("RAGAS BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:20}: {v}")
    print("=" * 60)

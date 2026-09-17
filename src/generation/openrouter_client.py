"""OpenRouter generation client for multi-model LLM completions."""


import time
from typing import Optional, List, Dict, Any
from openrouter import OpenRouter
from src.config import settings


class OpenRouterClient:
    """Wrapper around the official OpenRouter SDK supporting multi-model generation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None
    ):

        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.default_model = default_model or settings.LLM_MODEL

        if not self.api_key or self.api_key == "your_openrouter_api_key_here":
            raise ValueError(
                "OpenRouter API key is missing. Please set OPENROUTER_API_KEY in your .env file."
            )


    def generate(
        self,
        prompt: str,
        context: str = "",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate an answer given a prompt and retrieved context."""
        active_model = model or self.default_model

        if system_prompt is None:
            system_prompt = (
                "You are an expert AI assistant. Answer the user's question accurately "
                "using only the provided context. If the answer cannot be determined "
                "from the context, clearly state that you do not have enough information."
                "Do not generate answer that not in the provided context"
            )
        user_content = prompt
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            with OpenRouter(api_key=self.api_key) as client:
                response = client.chat.send(
                    model=active_model,
                    messages=messages,
                    temperature=temperature,
                )
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content or ""
                return "no content returned by model"
        except Exception as e:
            return f"[Error with model '{active_model}']: {e}"

    def generate_multi_model(
        self,
        prompt: str,
        context: str = "",
        models: Optional[List[str]] = None,
        temperature: float = 0.3,
    ) -> Dict[str, Dict[str, Any]]:
        """Query multiple models concurrently/sequentially and compare answers + latencies."""

        target_models = models or [
            "openai/gpt-4o-mini",
            "cohere/north-mini-code:free",
            "meta-llama/llama-3.3-70b-instruct",
        ]

        results: Dict[str, Dict[str, Any]] = {}

        for model_name in target_models:
            start_time = time.perf_counter()
            answer = self.generate(
                prompt=prompt,
                context=context,
                model=model_name,
                temperature=temperature,
            )
            elapsed = time.perf_counter() - start_time
            results[model_name] = {
                "answer": answer,
                "latency_sec": round(elapsed, 4),
            }
        return results





if __name__ == "__main__":
    client = OpenRouterClient()
    test_context = (
        "Agentic AI systems combine perception, memory retrieval, tool execution, "
        "and goal-directed planning to operate autonomously in complex environments."
    )
    test_question = "What are the essential building blocks of Agentic AI?"
    print("=" * 60)
    print("🤖 Multi-Model Generation Comparison via OpenRouter")
    print(f"Question: '{test_question}'")
    print("=" * 60)
    # Compare 3 different model families (OpenAI, Google Gemini, Meta Llama)
    benchmark_models = [
        "openai/gpt-4o-mini",
        "cohere/north-mini-code:free",
        "meta-llama/llama-3.3-70b-instruct",
    ]
    results = client.generate_multi_model(
        prompt=test_question,
        context=test_context,
        models=benchmark_models,
    )
    for model, data in results.items():
        print(f"\n🧠 Model: {model}")
        print(f"⏱️  Latency: {data['latency_sec']}s")
        print(f"💬 Answer:\n{data['answer']}")
        print("-" * 60)

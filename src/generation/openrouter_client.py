"""OpenRouter generation client for multi-model LLM completions."""


from typing import Optional, List, Dict
from openrouter import OpenRouter
from src.config import settings


class OpenRouterClient:
    """Wrapper around the official OpenRouter SDK for RAG generation."""
    
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
        with OpenRouter(api_key=self.api_key) as client:
            response = client.chat.send(
                model=active_model,
                messages=messages,
                temperature=temperature,
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            return ""


if __name__ == "__main__":
    client = OpenRouterClient()
    answer = client.generate(
        prompt="Explain what a vector database is in one sentence.",
        model="openai/gpt-4o-mini"
    )
    print(f"Generated Response:\n{answer}")
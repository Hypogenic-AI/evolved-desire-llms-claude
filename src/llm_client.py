"""
LLM Client for API calls.
Supports OpenAI and OpenRouter.
"""
import os
import time
from typing import Optional
from openai import OpenAI


class LLMClient:
    """Unified client for LLM API calls."""

    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize client.

        Args:
            provider: "openai" or "openrouter"
            model: Model name (defaults based on provider)
        """
        self.provider = provider

        if provider == "openai":
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.model = model or "gpt-4o-mini"
        elif provider == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY")
            )
            self.model = model or "openai/gpt-4o-mini"
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.total_tokens = 0
        self.call_count = 0

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
        retry_count: int = 3,
        retry_delay: float = 1.0
    ) -> str:
        """
        Get completion from LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            Response text
        """
        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                self.call_count += 1
                if response.usage:
                    self.total_tokens += response.usage.total_tokens

                return response.choices[0].message.content

            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise e

        return ""

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "provider": self.provider,
            "model": self.model,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens
        }


def test_client():
    """Quick test of client."""
    client = LLMClient(provider="openai")
    response = client.complete([
        {"role": "user", "content": "Say 'test successful' and nothing else."}
    ], temperature=0)
    print(f"Response: {response}")
    print(f"Stats: {client.get_stats()}")


if __name__ == "__main__":
    test_client()

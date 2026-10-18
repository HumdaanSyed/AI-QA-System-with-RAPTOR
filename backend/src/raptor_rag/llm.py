from collections.abc import Iterator
from typing import Protocol

from openai import OpenAI


class LLM(Protocol):
    model: str

    def stream(self, instructions: str, prompt: str) -> Iterator[str]:
        """Yield the answer text as it is generated."""
        ...

    def complete(self, instructions: str, prompt: str) -> str:
        """Return the whole answer text."""
        ...


class OpenAILLM:
    def __init__(self, api_key: str, model: str):
        # The SDK retries rate limits and server errors with backoff.
        self.client = OpenAI(api_key=api_key, max_retries=5)
        self.model = model

    def stream(self, instructions: str, prompt: str) -> Iterator[str]:
        with self.client.responses.stream(
            model=self.model,
            instructions=instructions,
            input=prompt,
            reasoning={"effort": "low"},  # grounded answers don't need long reasoning
        ) as events:
            for event in events:
                if event.type == "response.output_text.delta":
                    yield event.delta

    def complete(self, instructions: str, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model, instructions=instructions, input=prompt, reasoning={"effort": "low"}
        )
        return response.output_text

from collections.abc import Iterator
from typing import Protocol

from openai import OpenAI


class LLM(Protocol):
    def stream(self, instructions: str, prompt: str) -> Iterator[str]:
        """Yield the answer text as it is generated."""
        ...


class OpenAILLM:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
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

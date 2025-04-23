import json
import os
import re
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from loguru import logger


@dataclass
class ModelCompletion:
    generation: Dict[str, Any] | List[str]
    model: str


class Generation_Models(Enum):
    GPT4O = "gpt-4o-2024-08-06"
    GPT4O_MINI = "gpt-4o-mini-2024-07-18"
    CLAUDE_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_HAIKU = "claude-3-haiku-20240307"
    AZURE_GPT4O = "gpt-4o"  # "azure_ai/newgpt4o" #2024-02-15-preview
    TGI_GEMINI_9B = "gemma-2-9b-it"
    TOGETHER_LLAMA70B = "together_ai/meta-llama/Llama-3-70b-chat-hf"
    TOGETHER_GEMMA27B = "together_ai/google/gemma-2-27b-it"
    TOGETHER_GEMMA9B = "together_ai/google/gemma-2-9b-it"
    TOGETHER_LLAMA8B = "together_ai/meta-llama/Llama-3-8b-chat-hf"
    MOCK = "mock_llm"

    def __repr__(self):
        return self.value


class ModelProvider(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    TGI = "tgi"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    TOGETHER = "together"
    MOCK = "mock"


class BaseLLM:
    def __init__(self, model_name: Generation_Models, **kwargs) -> None:
        self.model_name = model_name

    def completion(
        self,
        prompt: List[Dict[str, str]] | List[List[Dict[str, str]]],
        structured_object: Any,
    ):
        NotImplementedError

    def _check_environment_variables(self):
        if self.model_provider == ModelProvider.OPENAI:
            assert os.getenv("OPENAI_API_KEY"), (
                "OPENAI_API_KEY environment variable not set"
            )
        elif self.model_provider == ModelProvider.ANTHROPIC:
            assert os.getenv("ANTHROPIC_API_KEY"), (
                "ANTHROPIC_API_KEY environment variable not set"
            )
        elif self.model_provider == ModelProvider.COHERE:
            assert os.getenv("COHERE_API_KEY"), (
                "COHERE_API_KEY environment variable not set"
            )
        elif self.model_provider == ModelProvider.AZURE:
            assert os.getenv("AZURE_API_KEY"), (
                "AZURE_API_KEY environment variable not set"
            )
            assert os.getenv("AZURE_API_BASE"), (
                "AZURE_API_BASE environment variable not set"
            )
        elif self.model_provider == ModelProvider.TGI:
            assert os.getenv("TGI_ENDPOINT"), (
                "TGI_ENDPOINT environment variable not set"
            )
        elif self.model_provider == ModelProvider.TOGETHER:
            assert os.getenv("TOGETHERAI_API_KEY"), (
                "TOGETHERAI_API_KEY environment variable not set"
            )
        elif self.model_provider == ModelProvider.MOCK:
            pass
        else:
            raise ValueError("Model provider not supported")

    @staticmethod
    def _maybe_sanitize_json(json_str: str) -> str | dict[str, Any]:
        try:
            return json.loads(re.sub(r"```json\s*|```", "", json_str))
        except json.JSONDecodeError:
            logger.warning("Incorrect JSON format returned")
            logger.debug(json_str)
            return json_str


class MockLLM(BaseLLM):
    def __init__(
        self,
        model_name: Generation_Models = Generation_Models.MOCK,
        model_provider: ModelProvider = ModelProvider.MOCK,
    ) -> None:
        super().__init__(model_name)

    def completion(
        self,
        prompt: List[Dict[str, str]] | List[List[Dict[str, str]]],
        structured_object: Optional[Any] = None,
    ) -> List[ModelCompletion]:
        assert isinstance(prompt, list), "Prompt must be a list"
        assert isinstance(prompt[0], dict) or isinstance(prompt[0], list), (
            "Prompt must be a list of dictionaries or a list of lists of dictionaries"
        )

        return [
            ModelCompletion(
                generation={"text": "Mock completion"}, model=self.model_name
            )
        ]


def _test():
    mock_llm = MockLLM()
    completions = mock_llm.completion(prompt=[{"text": "Hello"}])
    print(completions)


if __name__ == "__main__":
    _test()

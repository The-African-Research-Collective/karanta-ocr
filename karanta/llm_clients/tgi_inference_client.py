import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)

from post_training.llms.base import (
    BaseLLM,
    ModelCompletion,
    Generation_Models,
    ModelProvider,
)
from post_training.llms.utils import json_parse_model_output

load_dotenv()


class TGI_client(BaseLLM):
    def __init__(self, model_name: Generation_Models, model_provider: ModelProvider):
        super().__init__(model_name)
        self.model_provider = model_provider
        self._check_environment_variables()

        self.client = InferenceClient(os.getenv("TGI_ENDPOINT"))

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(3))
    async def completion(
        self,
        prompt: List[Dict[str, str]] | List[List[Dict[str, str]]],
        structured_object: Optional[Any] = None,
        **generation_kwargs: Any,
    ) -> List[ModelCompletion]:
        """ """
        temperature = generation_kwargs.get("temperature", 1.0)
        max_tokens = generation_kwargs.get("max_tokens", 512)

        def llm_inference(message):
            response = self.client.chat_completion(
                messages=message,
                seed=42,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            try:
                return json_parse_model_output(response.choices[0].message.content)
            except json.JSONDecodeError:
                return [{}]

        completions = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(llm_inference, message) for message in prompt]
            for future in futures:
                response = future.result()
                try:
                    completions.append(
                        ModelCompletion(
                            generation=response, model=self.model_name.value
                        )
                    )
                except json.JSONDecodeError:
                    raise ValueError("Error decoding response")

        return completions


async def _test():
    tgi_client = TGI_client(
        model_name=Generation_Models.TGI_GEMINI_9B, model_provider=ModelProvider.TGI
    )
    completions = await tgi_client.completion(
        prompt=[{"text": "Hello", "role": "user"}]
    )
    print(completions)


if __name__ == "__main__":
    asyncio.run(_test())

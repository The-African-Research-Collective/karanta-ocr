import asyncio
import os
import json
from functools import partial
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

import openai
from openai import AzureOpenAI
from pydantic import BaseModel
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

load_dotenv()


class AzureOPENAILLM(BaseLLM):
    def __init__(
        self,
        deployment_name: str,
        model_name: Generation_Models = Generation_Models.AZURE_GPT4O,
        model_provider: ModelProvider = ModelProvider.AZURE,
    ):
        super().__init__(model_name)
        self.model_provider = model_provider
        self._check_environment_variables()

        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )

        self.deployment_name = deployment_name

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(3))
    async def completion(
        self,
        prompt: List[Dict[str, str]] | List[List[Dict[str, str]]],
        structured_object: BaseModel | None = None,
        **generation_kwargs: Any,
    ) -> List[ModelCompletion]:
        """
        Generate completions for the given prompt using the model.
        """
        temperature = generation_kwargs.get("temperature", 1.0)
        max_tokens = generation_kwargs.get("max_tokens", 1024)

        if structured_object is not None:
            chat_func = partial(
                self.client.beta.chat.completions.parse,
                response_format=structured_object,
            )
        else:
            chat_func = self.client.chat.completions.create

        def llm_inference(message: List[Dict[str, str]]) -> dict[str, Any]:
            try:
                response = chat_func(
                    model=self.deployment_name,
                    messages=message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return self._maybe_sanitize_json(response.choices[0].message.content)
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return {}

        completions = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(llm_inference, message) for message in prompt]
            for future in futures:
                response = future.result()
                completions.append(
                    ModelCompletion(generation=response, model=self.model_name.value)
                )

        return completions


"""
Restoring the old class because of the error below:
response_format value as json_schema is enabled only for api versions 2024-08-01-preview and later
"""


class AzureOldDeployments(AzureOPENAILLM):
    def __init__(
        self,
        deployment_name: str,
        model_name: Generation_Models = Generation_Models.AZURE_GPT4O,
        model_provider: ModelProvider = ModelProvider.AZURE,
    ):
        super().__init__(deployment_name, model_name, model_provider)

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(3))
    async def completion(
        self,
        prompt: List[Dict[str, str]] | List[List[Dict[str, str]]],
        structured_object: Optional[Any] = None,
        **generation_kwargs: Any,
    ) -> List[ModelCompletion]:
        """
        Generate completions for the given prompt using the model.
        """
        temperature = generation_kwargs.get("temperature", 1.0)
        max_tokens = generation_kwargs.get("max_tokens", 1024)

        if structured_object:
            tools = [openai.pydantic_function_tool(structured_object)]
        else:
            tools = None

        def llm_inference(message: List[Dict[str, str]]):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=message,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if tools:
                    return json.loads(
                        response.choices[0].message.tool_calls[0].function.arguments
                    )
                else:
                    return response.choices[0].message.content
            except openai.BadRequestError:
                return {}

        completions = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(llm_inference, message) for message in prompt]
            for future in futures:
                response = future.result()
                completions.append(
                    ModelCompletion(generation=response, model=self.model_name.value)
                )

        return completions


async def _test():
    model = AzureOPENAILLM()
    prompt = [{"message": "Hello, how are you?", "role": "user"}]
    completions = await model.completion(prompt)
    print(completions)


if __name__ == "__main__":
    asyncio.run(_test())

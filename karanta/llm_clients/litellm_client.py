import json
import asyncio
import litellm
import warnings
from typing import Any, Dict, List, Optional

import openai
from dotenv import load_dotenv
from litellm import acompletion, batch_completion
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)

from karanta.llm_clients.utils import json_parse_model_output
from karanta.llm_clients.base import (
    BaseLLM,
    ModelCompletion,
    Generation_Models,
    ModelProvider,
)

load_dotenv()

litellm.suppress_debug_info = True
litellm.set_verbose = False

warnings.filterwarnings("ignore")


class LiteLLM(BaseLLM):
    def __init__(self, model_name: Generation_Models, model_provider: ModelProvider):
        super().__init__(model_name)
        self.model_provider = model_provider
        self._check_environment_variables()

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(10))
    async def completion(
        self,
        prompt: List[Dict[str, str]] | List[List[Dict[str, str]]],
        structured_object: Optional[Any] = None,
        **generation_kwargs: Any,
    ) -> List[ModelCompletion]:
        """ """
        temperature = generation_kwargs.get("temperature", 1.0)
        max_tokens = generation_kwargs.get("max_tokens", 512)

        completions = []

        try:
            if isinstance(prompt[0], list):
                response = batch_completion(
                    model=self.model_name.value,
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=structured_object,
                    drop_params=True,
                )

                for model_resp in response:
                    try:
                        if structured_object:
                            if self.model_provider == ModelProvider.TOGETHER:
                                completions.append(
                                    ModelCompletion(
                                        generation=json_parse_model_output(
                                            model_resp.choices[0].message.content
                                        ),
                                        model=self.model_name.value,
                                    )
                                )
                            else:
                                completions.append(
                                    ModelCompletion(
                                        generation=json.loads(
                                            model_resp.choices[0].message.content
                                        ),
                                        model=self.model_name.value,
                                    )
                                )
                        else:
                            completions.append(
                                ModelCompletion(
                                    generation=model_resp.choices[0].message.content,
                                    model=self.model_name.value,
                                )
                            )
                    except json.JSONDecodeError:
                        raise ValueError(
                            f"Error decoding response: {model_resp.choices[0].message.content}"
                        )

            elif isinstance(prompt[0], dict):
                response = await acompletion(
                    model=self.model_name.value,
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=structured_object,
                    drop_params=True,
                )

                try:
                    if structured_object:
                        if self.model_provider == ModelProvider.TOGETHER:
                            completions.append(
                                ModelCompletion(
                                    generation=json_parse_model_output(
                                        model_resp.choices[0].message.content
                                    ),
                                    model=self.model_name.value,
                                )
                            )
                        else:
                            completions.append(
                                ModelCompletion(
                                    generation=json.loads(
                                        model_resp.choices[0].message.content
                                    ),
                                    model=self.model_name.value,
                                )
                            )
                    else:
                        completions.append(
                            ModelCompletion(
                                generation=model_resp.choices[0].message.content,
                                model=self.model_name.value,
                            )
                        )
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Error decoding response: {response.choices[0].message.content}"
                    )

            else:
                raise ValueError(
                    "Prompt should be a list of dictionaries or a list of lists of dictionaries"
                )

        except openai.APITimeoutError as e:
            print(f"API Timeout Error: {e}")
            raise e
        except openai.APIConnectionError as e:
            print(f"API Connection Error: {e}")
            raise e
        except openai.RateLimitError as e:
            print(f"Rate Limit Error: {e}")
            raise e
        except Exception as e:
            print(f"Error: {e}")
            raise e

        return completions


async def _test():
    litellm: LiteLLM = LiteLLM(
        model_name=Generation_Models.GPT4O_MINI, model_provider=ModelProvider.OPENAI
    )
    completions = await litellm.completion(prompt=[{"text": "Hello", "role": "user"}])
    print(completions)


if __name__ == "__main__":
    asyncio.run(_test())

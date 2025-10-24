from enum import Enum

TARGET_IMAGE_DIM = 2048
PROMPT_PATH = "configs/prompts/open_ai_data_generation.yaml"
CREATE_TEST_PROMPT_PATH = "configs/prompts/create_tests.yaml"


class ModelGroup(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OLMO_VLLM = "olmo_vllm"


class Model(str, Enum):
    GPT_5 = "gpt-5"
    GPT_4O = "gpt-4o"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_CHAT_LATEST = "gpt-5-chat-latest"
    GPT_4_1 = "gpt-4.1-2025-04-14"
    GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"
    GPT_4_O = "gpt-4o-2024-08-06"
    OLMO_7B_PREVIEW = "allenai/olmOCR-7B-0225-preview"
    OLMO_7B_0725_FP8 = "allenai/olmOCR-7B-0725-FP8"
    OLMO_7B_0725 = "allenai/olmOCR-7B-0725"

from enum import Enum

TARGET_IMAGE_DIM = 2048
PROMPT_PATH = "configs/prompts/open_ai_data_generation.yaml"


class ModelGroup(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OLMO_VLLM = "olmo_vllm"


class Model(str, Enum):
    GPT_4_1 = "gpt-4.1-2025-04-14"
    GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"
    GPT_4_O = "gpt-4o-2024-08-06"
    OLMO_7B_PREVIEW = "allenai/olmOCR-7B-0225-preview"

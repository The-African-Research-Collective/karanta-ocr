"""
Code Adapted from: https://github.com/allenai/olmocr/blob/main/olmocr/bench/runners/run_olmocr_pipeline.py
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from karanta.pipeline import (
    MetricsKeeper,
    PageResult,
    WorkerTracker,
    vllm_server_host,
    vllm_server_ready,
    process_page,
)
from karanta.constants import PROMPT_PATH

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("karantaocr_runner")


# Basic configuration
@dataclass
class Args:
    model: str = "/home/oogundep/karanta-ocr/runs/karanta_set_qwen_2_5_3B_vl_all_samples_linear_no_base_text"
    server: str = "http://localhost:30044/v1"
    port: int = 30044
    max_model_len: int = 8192
    guided_decoding: bool = False
    gpu_memory_utilization: float = 0.8
    target_longest_image_dim: int = 1288
    target_anchor_text_len: int = 4000
    max_page_retries: int = 8
    max_page_error_rate: float = 0.004
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    prompt_key: str = "olmo_ocr_system_prompt"
    prompt_path: Optional[str] = PROMPT_PATH


server_check_lock = asyncio.Lock()


async def run_karanta_pipeline(
    pdf_path: str,
    page_num: int = 1,
    model: str = "/home/oogundep/karanta-ocr/runs/karanta_set_qwen_2_5_3B_vl_all_samples_linear_no_base_text",
) -> Optional[str]:
    """
    Process a single page of a PDF using the official olmocr pipeline's process_page function

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to process (1-indexed)

    Returns:
        The extracted text from the page or None if processing failed
    """
    # Ensure global variables are initialized
    global metrics, tracker
    if "metrics" not in globals() or metrics is None:
        metrics = MetricsKeeper(window=60 * 5)
    if "tracker" not in globals() or tracker is None:
        tracker = WorkerTracker()

    args = Args()
    args.model = model
    semaphore = asyncio.Semaphore(1)
    worker_id = 0  # Using 0 as default worker ID

    # Ensure server is running
    async with server_check_lock:
        _server_task = None
        try:
            await asyncio.wait_for(vllm_server_ready(args), timeout=5)
            logger.info("Using existing vllm server")
        except Exception:
            logger.info("Starting new vllm server")
            _server_task = asyncio.create_task(
                vllm_server_host(args.model, args, semaphore)
            )
            await vllm_server_ready(args)

    # Sets the model name used in the pipeline code, it's a hack sadly
    args.model = "karantaocr"

    try:
        # Process the page using the pipeline's process_page function
        # Note: process_page expects both original path and local path
        # In our case, we're using the same path for both
        page_result: PageResult = await process_page(
            args=args,
            worker_id=worker_id,
            pdf_orig_path=pdf_path,
            pdf_local_path=pdf_path,
            page_num=page_num,
        )

        # Return the natural text from the response
        if page_result and page_result.response and not page_result.is_fallback:
            return page_result.response.pages[0].natural_text
        return None

    except Exception as e:
        logger.error(f"Error processing page: {type(e).__name__} - {str(e)}")
        return None

    finally:
        # We leave the server running for potential reuse
        pass


async def main():
    # Example usage
    pdf_path = "/home/oogundep/karanta_bench/karanta-ocr/runs/olmOCR-bench/bench_data/pdfs/old_scans_math/1_pg10.pdf"
    page_num = 1

    result = await run_karanta_pipeline(pdf_path, page_num)
    if result:
        print(f"Extracted text: {result[:200]}...")  # Print first 200 chars
    else:
        print("Failed to extract text from the page")


if __name__ == "__main__":
    asyncio.run(main())

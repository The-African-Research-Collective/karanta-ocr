"""
VLLM Client for batch inference workers
"""

import time
import logging
from typing import Dict, List, Optional, Any
from openai import OpenAI
import requests

logger = logging.getLogger(__name__)


def openai_response_format_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "page_response",
            "schema": {
                "type": "object",
                "properties": {
                    "primary_language": {
                        "type": ["string", "null"],
                        "description": "The primary language of the text using two-letter codes or null if there is no text at all that you think you should read.",
                    },
                    "is_rotation_valid": {
                        "type": "boolean",
                        "description": "Is this page oriented correctly for reading? Answer only considering the textual content, do not factor in the rotation of any charts, tables, drawings, or figures.",
                    },
                    "rotation_correction": {
                        "type": "integer",
                        "description": "Indicates the degree of clockwise rotation needed if the page is not oriented correctly.",
                        "enum": [0, 90, 180, 270],
                        "default": 0,
                    },
                    "is_table": {
                        "type": "boolean",
                        "description": "Indicates if the majority of the page content is in tabular format.",
                    },
                    "is_diagram": {
                        "type": "boolean",
                        "description": "Indicates if the majority of the page content is a visual diagram.",
                    },
                    "natural_text": {
                        "type": ["string", "null"],
                        "description": "The natural text content extracted from the page.",
                    },
                },
                "additionalProperties": False,
                "required": [
                    "primary_language",
                    "is_rotation_valid",
                    "rotation_correction",
                    "is_table",
                    "is_diagram",
                    "natural_text",
                ],
            },
            "strict": True,
        },
    }


class VLLMClientError(Exception):
    """Custom exception for VLLM client errors"""

    pass


class VLLMClient:
    """
    VLLM Client that uses OpenAI-compatible API interface

    This client manages connections to VLLM servers and provides
    a unified interface for text generation tasks.
    """

    def __init__(
        self,
        port: int,
        host: str = "localhost",
        api_key: str = "EMPTY",
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        health_check_timeout: float = 30.0,
    ):
        """
        Initialize VLLM client

        Args:
            port: VLLM server port
            host: VLLM server host (default: localhost)
            api_key: API key for authentication (default: EMPTY for VLLM)
            timeout: Request timeout in seconds (default: 300.0)
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            health_check_timeout: Timeout for health checks in seconds
        """
        self.port = port
        self.host = host
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.health_check_timeout = health_check_timeout

        # Build server URL
        self.base_url = f"http://{self.host}:{self.port}/v1"
        self.health_url = f"http://{self.host}:{self.port}/health"

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        # Server info cache
        self._server_info = None
        self._last_health_check = 0
        self._health_check_interval = 60  # Check health every 60 seconds

        logger.info(f"Initialized VLLM client for {self.base_url}")

    def health_check(self, force: bool = False) -> bool:
        """
        Check if VLLM server is healthy

        Args:
            force: Force health check even if recently checked

        Returns:
            True if server is healthy, False otherwise
        """
        current_time = time.time()

        # Skip health check if recently performed (unless forced)
        if (
            not force
            and (current_time - self._last_health_check) < self._health_check_interval
        ):
            return True

        try:
            response = requests.get(self.health_url, timeout=self.health_check_timeout)

            is_healthy = response.status_code == 200
            self._last_health_check = current_time

            if not is_healthy:
                logger.warning(
                    f"VLLM server health check failed: {response.status_code}"
                )

            return is_healthy

        except requests.RequestException as e:
            logger.error(f"VLLM server health check failed: {e}")
            return False

    def get_server_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get server information (models, version, etc.)

        Args:
            force_refresh: Force refresh of cached server info

        Returns:
            Dictionary containing server information
        """
        if self._server_info is None or force_refresh:
            try:
                # Get models list
                models_response = self.client.models.list()
                models = [model.id for model in models_response.data]

                # Try to get additional server info
                server_info = {
                    "models": models,
                    "base_url": self.base_url,
                    "port": self.port,
                    "host": self.host,
                    "last_updated": time.time(),
                }

                # Try to get version info from health endpoint
                try:
                    health_response = requests.get(self.health_url, timeout=5)
                    if health_response.status_code == 200:
                        health_data = health_response.json()
                        server_info.update(health_data)
                except Exception:
                    pass  # Health endpoint might not return JSON

                self._server_info = server_info
                logger.info(f"Retrieved server info: {len(models)} models available")

            except Exception as e:
                logger.error(f"Failed to get server info: {e}")
                raise VLLMClientError(f"Failed to get server info: {e}")

        return self._server_info

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text using VLLM server

        Args:
            prompt: Input prompt for generation
            model: Model name (if None, uses first available model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            stream: Whether to stream response
            **kwargs: Additional parameters passed to OpenAI API

        Returns:
            Dictionary containing generated text and metadata
        """
        # Health check before generation
        if not self.health_check():
            raise VLLMClientError(f"VLLM server at {self.base_url} is not healthy")

        # Get model if not specified
        if model is None:
            server_info = self.get_server_info()
            if not server_info.get("models"):
                raise VLLMClientError("No models available on server")
            model = server_info["models"][0]
            logger.debug(f"Using default model: {model}")

        # Prepare generation parameters
        generation_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": response_format,
            **kwargs,
        }

        # Generate with retries
        last_exception = None
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Generation attempt {attempt + 1}/{self.max_retries + 1}")

                # Make API call
                response = self.client.chat.completions.create(**generation_params)

                # Process response
                return self._process_response(response, start_time, generation_params)

            except Exception as e:
                last_exception = e
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))  # Exponential backoff
                    continue
                else:
                    break

        # All retries failed
        error_msg = f"Generation failed after {self.max_retries + 1} attempts. Last error: {last_exception}"
        logger.error(error_msg)
        raise VLLMClientError(error_msg)

    def _process_response(
        self, response, start_time: float, generation_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process non-streaming response"""
        end_time = time.time()

        if not response.choices:
            raise VLLMClientError("No choices returned from VLLM server")

        choice = response.choices[0]

        result = {
            "text": choice.message.content,
            "finish_reason": choice.finish_reason,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens
                if response.usage
                else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            "metadata": {
                "generation_time": end_time - start_time,
                "server_url": self.base_url,
                "generation_params": {
                    k: v
                    for k, v in generation_params.items()
                    if k not in ["messages"]  # Exclude potentially large prompt
                },
                "timestamp": end_time,
            },
        }

        logger.debug(
            f"Generation completed in {result['metadata']['generation_time']:.2f}s"
        )
        return result

    def batch_generate(
        self, prompts: List[str], **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts

        Args:
            prompts: List of prompts to process
            **generation_kwargs: Arguments passed to generate()

        Returns:
            List of generation results
        """
        results = []

        for i, prompt in enumerate(prompts):
            try:
                logger.debug(f"Processing prompt {i + 1}/{len(prompts)}")
                result = self.generate(prompt, **generation_kwargs)
                result["metadata"]["batch_index"] = i
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process prompt {i + 1}: {e}")
                results.append(
                    {"error": str(e), "metadata": {"batch_index": i, "failed": True}}
                )

        return results

    def __repr__(self) -> str:
        return (
            f"VLLMClient(host={self.host}, port={self.port}, base_url={self.base_url})"
        )


class VLLMClientManager:
    """
    Manager class for multiple VLLM clients
    Handles client creation and routing based on worker configuration
    """

    def __init__(self, server_config: Dict[int, str] = None):
        """
        Initialize client manager

        Args:
            server_config: Dictionary mapping ports to hosts
                          e.g., {8000: "localhost", 8001: "gpu-node-1"}
        """
        self.server_config = server_config or {}
        self.clients: Dict[int, VLLMClient] = {}

    def get_client(self, port: int, **client_kwargs) -> VLLMClient:
        """
        Get or create VLLM client for given port

        Args:
            port: VLLM server port
            **client_kwargs: Additional arguments for VLLMClient

        Returns:
            VLLMClient instance
        """
        if port not in self.clients:
            host = self.server_config.get(port, "localhost")

            self.clients[port] = VLLMClient(port=port, host=host, **client_kwargs)

            logger.info(f"Created VLLM client for port {port}")

        return self.clients[port]

    def health_check_all(self) -> Dict[int, bool]:
        """
        Check health of all managed clients

        Returns:
            Dictionary mapping ports to health status
        """
        results = {}

        for port, client in self.clients.items():
            try:
                results[port] = client.health_check(force=True)
            except Exception as e:
                logger.error(f"Health check failed for port {port}: {e}")
                results[port] = False

        return results

    def get_client_from_worker_name(
        self, worker_name: str, **client_kwargs
    ) -> VLLMClient:
        """
        Extract port from worker name and return appropriate client

        Worker name format: worker_port_{port}_{worker_index}

        Args:
            worker_name: Celery worker hostname
            **client_kwargs: Additional arguments for VLLMClient

        Returns:
            VLLMClient instance
        """
        try:
            # Extract port from worker name
            # Format: worker_port_8000_1@hostname
            parts = worker_name.split("_")
            port_index = parts.index("port")
            port = int(parts[port_index + 1])

            return self.get_client(port, **client_kwargs)

        except (ValueError, IndexError):
            raise VLLMClientError(
                f"Invalid worker name format: {worker_name}. Expected format: worker_gpu_{{gpu_id}}_port_{{port}}_{{worker_index}}@hostname"
            )


# Global client manager instance
client_manager = VLLMClientManager()


def get_vllm_client_for_worker(worker_name: str, **kwargs) -> VLLMClient:
    """
    Convenience function to get VLLM client for current worker

    Args:
        worker_name: Current worker hostname
        **kwargs: Additional arguments for VLLMClient

    Returns:
        VLLMClient instance
    """
    return client_manager.get_client_from_worker_name(worker_name, **kwargs)


# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    import logging

    logging.basicConfig(level=logging.INFO)

    # Test basic client
    try:
        client = VLLMClient(port=8005)

        # Health check
        if client.health_check():
            print("✓ Server is healthy")

            # Get server info
            info = client.get_server_info()
            print(f"✓ Available models: {info.get('models', [])}")

            # Generate text
            result = client.generate(
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                max_tokens=50,
                temperature=0.7,
            )

            print(f"✓ Generated text: {result['text']}")
            print(f"✓ Generation time: {result['metadata']['generation_time']:.2f}s")

        else:
            print("✗ Server health check failed")

    except Exception as e:
        print(f"✗ Error: {e}")

    # Test client manager
    try:
        manager = VLLMClientManager({8005: "localhost", 8006: "localhost"})

        # Test worker name parsing
        worker_name = "worker_port_8006_1@hostname"
        client = manager.get_client_from_worker_name(worker_name)

        # Get server info
        print(f"✓ Created client for worker {worker_name}: {client}")

        # Get server info
        info = client.get_server_info()
        print(f"✓ Available models: {info.get('models', [])}")

        health_check_results = manager.health_check_all()
        print(f"✓ Health check results: {health_check_results}")

    except Exception as e:
        print(f"✗ Manager error: {e}")

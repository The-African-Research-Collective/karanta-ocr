import importlib
import json
import sys
from typing import Any, Dict

from pydantic import BaseModel


# TODO: @theyorubayesian - Allow containers (e.g. list, dict, None)
def get_response_format_for_model(response_structure: str) -> type[BaseModel]:
    sys.path.insert(0, ".")

    module_path, class_name = response_structure.rsplit(".", 1)
    module = importlib.import_module(module_path)
    response_class = getattr(module, class_name)

    assert issubclass(response_class, BaseModel), (
        f"The response structure '{response_structure}' is not a subclass of Pydantic BaseModel."
    )

    return response_class


def json_parse_model_output(output: str) -> Dict[str, Any]:
    """
    This function parses the output of a model and returns a dictionary.
    it works by finding the first opening bracket and removing everything before it.
    Then it finds the last closing bracket and removes everything after it.
    Finally, it returns the JSON object.
    """

    # Find the first opening bracket and remove everything before it
    start = output.find("[")
    start_curly = output.find("{")

    if (start != -1 and start_curly < start) or (start == -1 and start_curly != -1):
        start = start_curly
        end = output.rfind("}")
    else:
        end = output.rfind("]", start)

    output = output[start:]
    output = output[: end - start + 1]

    return json.loads(output)


CHAT_TEMPLATES = []

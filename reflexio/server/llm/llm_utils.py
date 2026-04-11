import inspect
from typing import Any

from pydantic import BaseModel


def is_pydantic_model(response_format: Any) -> bool:
    """
    Check if response_format is a Pydantic BaseModel class.

    Args:
        response_format: Response format to check.

    Returns:
        True if response_format is a Pydantic BaseModel class, False otherwise.
    """
    return inspect.isclass(response_format) and issubclass(response_format, BaseModel)

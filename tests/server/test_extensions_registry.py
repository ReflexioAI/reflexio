import pytest
from reflexio.server.extensions import (
    ServiceKey, register_service, get_service, require_service, reset_services,
)

KEY: ServiceKey[str] = ServiceKey("demo")


def setup_function() -> None:
    reset_services()


def test_get_returns_none_when_unset() -> None:
    assert get_service(KEY) is None


def test_register_then_get_and_require() -> None:
    register_service(KEY, "x")
    assert get_service(KEY) == "x"
    assert require_service(KEY) == "x"


def test_require_raises_when_unset() -> None:
    with pytest.raises(LookupError):
        require_service(KEY)


def test_double_register_is_error_unless_override() -> None:
    register_service(KEY, "x")
    with pytest.raises(RuntimeError):
        register_service(KEY, "y")
    register_service(KEY, "y", override=True)
    assert get_service(KEY) == "y"

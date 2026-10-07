from app import handle_turn
from support import Model, sdk_client


def test_existing_response():
    assert (
        handle_turn(
            sdk_client(), Model(), user_message="hello", user_id="u1", session_id="s1"
        )
        == "response:hello"
    )

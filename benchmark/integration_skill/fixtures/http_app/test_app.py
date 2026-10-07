from app import handle_turn
from support import HTTP, Model


def test_existing_response():
    assert (
        handle_turn(
            HTTP(), Model(), user_message="hello", user_id="u1", session_id="s1"
        )
        == "response:hello"
    )

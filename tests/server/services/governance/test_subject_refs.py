from reflexio.server.services.governance.subject_refs import (
    actor_ref,
    request_ref,
    stable_id,
    subject_ref,
)


def test_refs_are_deterministic_prefixed_and_do_not_leak_raw_values():
    secret = "test-secret"

    sub = subject_ref("alice@example.com", secret)
    actor = actor_ref("api-token-name", secret)
    req = request_ref("request-id-with-alice@example.com", secret)

    assert sub == subject_ref("alice@example.com", secret)
    assert sub.startswith("subref_v1_")
    assert actor.startswith("actref_v1_")
    assert req.startswith("reqref_v1_")
    assert "alice" not in sub
    assert "example.com" not in sub
    assert "api-token-name" not in actor
    assert "request-id" not in req


def test_stable_id_is_deterministic_and_prefixed():
    first = stable_id("purge", "org1:user_erasure:subref:reqref")
    second = stable_id("purge", "org1:user_erasure:subref:reqref")

    assert first == second
    assert first.startswith("purge_")

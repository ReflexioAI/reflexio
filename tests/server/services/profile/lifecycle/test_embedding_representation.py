"""Semantic profile text omits only operational lifecycle bookkeeping."""

from copy import deepcopy

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.server.services.embedding_text import embedding_input, embedding_text
from reflexio.server.services.profile.lifecycle import (
    KEY,
    attention_context,
    attention_epoch,
    metadata,
)
from reflexio.server.services.profile.lifecycle_deadlines import job_for

from .test_lifecycle import DEADLINE, T0, profile


def test_equivalent_fact_has_identical_embedding_input_without_lifecycle_bookkeeping():
    fact = profile()
    plain = fact.model_copy(update={"custom_features": None})
    assert embedding_text(fact) == embedding_text(plain) == fact.content
    for model in [
        "local/nomic-embed-text-v1.5",
        "local/multilingual-e5-small",
        "text-embedding-3-small",
    ]:
        assert embedding_input(
            embedding_text(fact), model_name=model
        ) == embedding_input(embedding_text(plain), model_name=model)
        assert embedding_input(
            embedding_text(fact), model_name=model, purpose="query"
        ) == embedding_input(embedding_text(plain), model_name=model, purpose="query")


def test_ordinary_features_survive_exactly_in_order_and_nested_lookalikes_are_retained():
    ordinary = {
        "theme": "dark",
        "profile_lifecycle_v1_detail": "meaningful ordinary feature",
        "nested": {KEY: "ordinary nested text"},
    }
    fact = profile()
    assert fact.custom_features is not None
    lifecycle = deepcopy(fact.custom_features)
    fact.custom_features = {
        "theme": ordinary["theme"],
        **lifecycle,
        **{key: value for key, value in ordinary.items() if key != "theme"},
    }
    expected = fact.content + "\n" + str(ordinary)
    assert embedding_text(fact) == expected
    assert (
        embedding_text(fact.model_copy(update={"custom_features": ordinary}))
        == expected
    )


def test_embedding_projection_does_not_mutate_any_persisted_control_authority():
    fact = profile(deadline=DEADLINE)
    assert fact.custom_features is not None
    fact.custom_features[KEY]["ancestor_ids"] = ["old"]
    fact.custom_features[KEY]["explicit_forget_quote"] = (
        "source-backed forget instruction"
    )
    before = deepcopy(fact.model_dump(mode="json"))
    before_job = job_for(before)
    assert before_job is not None
    for _ in range(2):
        assert embedding_text(fact) == fact.content
    assert fact.model_dump(mode="json") == before
    assert metadata(fact)["explicit_storage_deadline"] == DEADLINE
    assert metadata(fact)["explicit_forget_quote"] == "source-backed forget instruction"
    assert job_for(fact.model_dump(mode="json")) == before_job
    with attention_context({"old": T0 + 10}):
        assert attention_epoch(fact, now=T0 + 20) == T0 + 10


def test_bookkeeping_updates_do_not_change_semantic_input_or_strip_literal_fact_text():
    fact = profile()
    fact.content = f"User works on {KEY} diagnostics."
    before_text = embedding_text(fact)
    assert KEY in before_text  # the content itself is never scrubbed
    assert fact.custom_features is not None
    fact.custom_features[KEY]["created_at"] += 10
    fact.custom_features[KEY]["ancestor_ids"] = ["one", "two"]
    fact.custom_features[KEY]["explicit_storage_deadline"] = DEADLINE
    assert embedding_text(fact) == before_text == fact.content


def test_other_entity_text_is_unchanged_even_when_it_mentions_the_reserved_name():
    interaction = Interaction(
        interaction_id=1,
        user_id="u",
        request_id="r",
        role="user",
        content=f"Discuss {KEY}",
        user_action_description="clicked lifecycle help",
    )
    assert embedding_text(interaction) == f"Discuss {KEY}\nclicked lifecycle help"

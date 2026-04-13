"""
Canned query set and vocabulary for the retrieval latency benchmark.

The query set, vocabulary, and templates are frozen as module-level
constants so that benchmark runs are bit-for-bit reproducible across
machines and across time. Do not mutate.
"""

from __future__ import annotations

# 20 queries spanning short / long / keyword-heavy / semantic-heavy.
# Query diversity matters because hybrid search behavior depends on
# how many FTS tokens match vs how well the query vector aligns.
QUERIES: tuple[str, ...] = (
    # short, keyword-heavy
    "refund policy",
    "shipping delay",
    "cancel subscription",
    "password reset",
    "billing error",
    # short, semantic
    "user was frustrated with the agent",
    "agent apologized and offered compensation",
    "customer wants to speak to a manager",
    "sales opportunity upsell",
    "technical issue escalation",
    # long, conversational
    "the user complained that the previous agent had been rude and dismissive during a prior call",
    "looking for playbooks where we recommend offering a discount when a customer threatens to churn",
    "anything related to the enterprise plan negotiation playbook for high-value accounts",
    "profiles of users who have reported repeated shipping problems in the last month",
    # long, semantic
    "what do we know about users who rely on the mobile app and have poor network conditions",
    "situations where the agent should not offer a refund because policy prevents it",
    "user asks about product roadmap and pricing changes",
    # keyword-heavy, rare tokens
    "api key rotation quota limit",
    "webhook signature verification failure",
    "gdpr data export request",
)


# Fixed small vocabulary used to build synthetic profiles and playbooks.
# Enough diversity for non-trivial BM25 scoring but not large enough to
# dominate memory at N=10_000.
VOCABULARY: tuple[str, ...] = (
    "refund",
    "shipping",
    "cancel",
    "subscription",
    "password",
    "billing",
    "frustrated",
    "apologized",
    "manager",
    "sales",
    "technical",
    "escalation",
    "rude",
    "dismissive",
    "discount",
    "churn",
    "enterprise",
    "negotiation",
    "mobile",
    "network",
    "roadmap",
    "pricing",
    "policy",
    "quota",
    "webhook",
    "gdpr",
    "export",
    "verification",
    "signature",
    "rotation",
    "limit",
    "profile",
    "playbook",
    "agent",
    "user",
    "request",
    "session",
    "context",
    "history",
    "interaction",
    "feedback",
    "resolution",
    "complaint",
    "praise",
    "upgrade",
    "downgrade",
    "trial",
    "payment",
    "invoice",
    "receipt",
    "tax",
    "shipping",
    "delivery",
    "tracking",
    "return",
    "exchange",
    "warranty",
    "bug",
    "crash",
    "error",
    "timeout",
    "latency",
    "performance",
    "slow",
    "fast",
    "reliable",
    "consistent",
    "intermittent",
    "regression",
    "feature",
)


# Templated sentences used to build synthetic playbook/profile content.
# Each template has at least one `{word}` slot so the synthetic text has
# real token variety across the corpus.
CONTENT_TEMPLATES: tuple[str, ...] = (
    "The user asked about {word} and the agent should clarify policy details.",
    "When the customer mentions {word}, confirm the account status first.",
    "Escalate to a senior agent if the user repeats {word} after one attempt.",
    "Offer a {word} to retain accounts showing churn risk indicators.",
    "Capture the {word} and log it against the user's profile for follow-up.",
    "The agent should proactively mention {word} during enterprise calls.",
    "If {word} is unclear, ask a clarifying question before committing.",
    "Document the {word} in the session notes with a timestamp.",
    "Do not promise {word} without checking the internal knowledge base.",
    "Recommend {word} as the next step when the user sounds uncertain.",
)

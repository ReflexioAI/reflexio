import logging

logger = logging.getLogger(__name__)

# Shared prefix for aggregate lineage event reasons.
# Consumers: storage_base (here), sqlite_storage/_playbook.py, and lib/_agent_playbook.py
# (which imports this constant to keep the parser and producers in sync).
AGGREGATE_REASON_PREFIX = "aggregate:"

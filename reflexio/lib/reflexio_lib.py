"""Reflexio facade — assembled from domain mixins."""

from reflexio.lib._agent_playbook import AgentPlaybookMixin
from reflexio.lib._config import ConfigMixin
from reflexio.lib._dashboard import DashboardMixin
from reflexio.lib._generation import GenerationMixin
from reflexio.lib._interactions import InteractionsMixin
from reflexio.lib._operations import OperationsMixin
from reflexio.lib._profiles import ProfilesMixin
from reflexio.lib._search import SearchMixin
from reflexio.lib._user_playbook import UserPlaybookMixin


class Reflexio(
    InteractionsMixin,
    ProfilesMixin,
    AgentPlaybookMixin,
    UserPlaybookMixin,
    ConfigMixin,
    GenerationMixin,
    OperationsMixin,
    DashboardMixin,
    SearchMixin,
):
    """Synchronous facade providing a unified API for all Reflexio operations."""

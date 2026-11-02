"""Bootstrap policy imports must work in stripped inference distributions."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_historical_pending_conversion_and_diagnostic_do_not_require_models():
    code = """
import sys
class RejectModelSchemas:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'reflexio.models' or fullname.startswith('reflexio.models.'):
            raise AssertionError('Bootstrap attempted model-schema import: ' + fullname)
sys.meta_path.insert(0, RejectModelSchemas())
from reflexio import pending_tool_policy
assert not hasattr(pending_tool_policy, 'resolve_resume_timing')
assert pending_tool_policy.strip_legacy_pending_tool_timing({
    'enabled': False, 'resume_poll_interval_seconds': 0.25,
    'resume_claim_ttl_seconds': 1800,
}) == {'enabled': False}
pending_tool_policy.warn_retired_resume_timing({})
assert not any(name.startswith('reflexio.models') for name in sys.modules)
"""
    result = subprocess.run(  # noqa: S603 - fixed owned Python program
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr

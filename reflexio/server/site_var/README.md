# Site Variables
Description: Global configuration manager for site-wide variables and per-org feature flags

## Main Entry Points

- **Manager**: `site_var_manager.py` - `SiteVarManager` (singleton)
- **Feature Flags**: `feature_flags.py` - Per-org feature gating helpers
- **Sources**: `site_var_sources/` - JSON/TXT config files

## Purpose

1. **Global settings** - Model names, embedding models
2. **Feature flags** - Per-org feature gating (global enable or per-org allowlist)
3. **Dual storage** - File-based with optional Redis caching
4. **Auto-fallback** - Redis → File system graceful degradation

## Feature Flags

**File**: `feature_flags.py`
**Config**: `site_var_sources/feature_flags.json`

Per-org feature gating with fail-open defaults. Each flag supports global enable or per-org allowlist.

```python
from reflexio.server.site_var.feature_flags import is_feature_enabled, get_all_feature_flags

is_feature_enabled(org_id, "deduplicator")  # Single flag check
get_all_feature_flags(org_id)               # All flags as dict[str, bool]
```

**Config format** (`feature_flags.json`):
```json
{
    "deduplicator": {
        "enabled": true,
        "enabled_org_ids": []
    }
}
```

**Resolution logic**: Enabled if `enabled=True` (global) OR `org_id` in `enabled_org_ids`. Unknown flags default to enabled (fail-open).

**Current flags**: `invitation_only` (global, gates registration), `deduplicator` (gates playbook deduplication).

## Usage

```python
from reflexio.server.site_var.site_var_manager import SiteVarManager

manager = SiteVarManager()
config = manager.get_site_var("app_config")  # Returns dict or string
```

## File Structure

```
site_var/
├── site_var_manager.py        # SiteVarManager (singleton)
├── feature_flags.py           # Per-org feature flag helpers
└── site_var_sources/
    ├── app_config.json        # JSON → parsed dict
    ├── model_config.json
    └── feature_flags.json     # Feature flag config (per-flag enable + org allowlist)
```

## Architecture Pattern

- **JSON priority**: `.json` files take precedence over `.txt`
- **Variable name**: Filename without extension
- **Redis optional**: Enable via `SiteVarManager(enable_redis=True)`
- **Feature flags**: Loaded via SiteVarManager, resolved per-org at request time

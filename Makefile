# Release automation for reflexio-ai (PyPI).
#
# Usage:
#   make bump VERSION=0.2.16     Update version in pyproject.toml + refresh uv.lock
#   make release VERSION=0.2.16  Bump, commit, tag v0.2.16, publish to PyPI, push
#   make publish                 Build and publish current version to PyPI
#   make publish-dry             Build only; inspect dist/ without uploading
#   make test-pypi               Build and publish current version to TestPyPI
#   make client-release VERSION=0.2.16
#                                Bump, test, tag, and publish reflexio-client only
#
# Requires:
#   - uv (UV_PUBLISH_TOKEN set for PyPI uploads, or ~/.pypirc)
#   - git (for the release flow)

.PHONY: help bump release publish publish-dry test-pypi clean version \
        check-version check-clean check-branch check-up-to-date \
        check-tag-free verify-dist \
        client-version check-client-version check-client-tag-free \
        client-clean client-bump client-test client-publish \
        client-publish-dry client-test-pypi client-release \
        verify-client-dist

PYPROJECT := pyproject.toml
CLIENT_DIR := client_dist
CLIENT_PYPROJECT := $(CLIENT_DIR)/pyproject.toml
VERSION_CURRENT := $(shell grep -E '^version = ' $(PYPROJECT) | head -1 | cut -d'"' -f2)
CLIENT_VERSION_CURRENT := $(shell grep -E '^version = ' $(CLIENT_PYPROJECT) | head -1 | cut -d'"' -f2)

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/{printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

version: ## Print the current package version
	@echo $(VERSION_CURRENT)

client-version: ## Print the current reflexio-client package version
	@echo $(CLIENT_VERSION_CURRENT)

check-version:
ifndef VERSION
	$(error VERSION is required, e.g. make bump VERSION=0.2.16)
endif
	@printf '%s' '$(VERSION)' | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$$' \
	  || { echo "error: VERSION '$(VERSION)' is not valid semver" >&2; exit 1; }

check-clean:
	@git diff --quiet && git diff --cached --quiet \
	  || { echo "error: working tree is dirty — commit or stash first" >&2; exit 1; }

check-branch:
	@branch=$$(git rev-parse --abbrev-ref HEAD); \
	  [ "$$branch" = "main" ] || { echo "error: not on main (on $$branch)" >&2; exit 1; }

check-up-to-date:
	@git fetch origin
	@[ -z "$$(git log HEAD..origin/main --oneline)" ] \
	  || { echo "error: local main is behind origin/main — pull first" >&2; exit 1; }

check-tag-free: check-version
	@if git rev-parse -q --verify "refs/tags/v$(VERSION)" >/dev/null; then \
	  echo "error: tag v$(VERSION) already exists locally" >&2; exit 1; fi
	@if git ls-remote --exit-code --tags origin "refs/tags/v$(VERSION)" >/dev/null 2>&1; then \
	  echo "error: tag v$(VERSION) already exists on origin" >&2; exit 1; fi

check-client-version:
ifndef VERSION
	$(error VERSION is required, e.g. make client-release VERSION=0.2.16)
endif
	@printf '%s' '$(VERSION)' | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$$' \
	  || { echo "error: VERSION '$(VERSION)' is not valid semver" >&2; exit 1; }

check-client-tag-free: check-client-version
	@if git rev-parse -q --verify "refs/tags/reflexio-client-v$(VERSION)" >/dev/null; then \
	  echo "error: tag reflexio-client-v$(VERSION) already exists locally" >&2; exit 1; fi
	@if git ls-remote --exit-code --tags origin "refs/tags/reflexio-client-v$(VERSION)" >/dev/null 2>&1; then \
	  echo "error: tag reflexio-client-v$(VERSION) already exists on origin" >&2; exit 1; fi

verify-dist:
	@actual=$$(grep -E '^version = ' $(PYPROJECT) | head -1 | cut -d'"' -f2); \
	  ls dist/*-$$actual* >/dev/null 2>&1 \
	    || { echo "error: dist/ has no artifacts for version $$actual" >&2; exit 1; }; \
	  echo "✓ dist/ contains artifacts for version $$actual"

verify-client-dist:
	@actual=$$(grep -E '^version = ' $(CLIENT_PYPROJECT) | head -1 | cut -d'"' -f2); \
	  ls $(CLIENT_DIR)/dist/*-$$actual* >/dev/null 2>&1 \
	    || { echo "error: $(CLIENT_DIR)/dist/ has no artifacts for version $$actual" >&2; exit 1; }; \
	  echo "✓ $(CLIENT_DIR)/dist/ contains artifacts for version $$actual"

clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info

client-clean: ## Remove reflexio-client build artifacts
	rm -rf $(CLIENT_DIR)/dist/ $(CLIENT_DIR)/build/ $(CLIENT_DIR)/*.egg-info

bump: check-version check-clean ## Rewrite version in pyproject.toml and refresh uv.lock
	@echo "→ bumping to $(VERSION)"
	@sed -i.bak -E 's/^version = "[^"]+"/version = "$(VERSION)"/' $(PYPROJECT)
	@rm -f $(PYPROJECT).bak
	@echo "→ refreshing uv lockfile"
	@uv lock
	@echo "→ resulting version:"
	@grep -E '^version = ' $(PYPROJECT)

client-bump: check-client-version check-clean ## Rewrite reflexio-client version and refresh its uv.lock
	@echo "→ bumping reflexio-client to $(VERSION)"
	@sed -i.bak -E 's/^version = "[^"]+"/version = "$(VERSION)"/' $(CLIENT_PYPROJECT)
	@rm -f $(CLIENT_PYPROJECT).bak
	@echo "→ refreshing reflexio-client uv lockfile"
	cd $(CLIENT_DIR) && uv lock
	@echo "→ resulting reflexio-client version:"
	@grep -E '^version = ' $(CLIENT_PYPROJECT)

client-test: ## Run client-only lint, type checks, and tests
	uv run ruff check reflexio/client reflexio/models tests/client tests/models
	uv run pyright reflexio/client reflexio/models tests/client tests/models
	uv run pytest tests/client tests/models -q -o addopts=

publish: clean ## Build and publish current version to PyPI
	@echo "→ uv build"
	uv build
	@$(MAKE) verify-dist
	@echo "→ uv publish"
	uv publish dist/*

publish-dry: clean ## Build only; show what would ship without uploading
	@echo "→ uv build (dry: inspect dist/ manually)"
	uv build
	@ls -la dist/

test-pypi: clean ## Build and publish current version to TestPyPI
	@echo "→ uv build"
	uv build
	@echo "→ uv publish --publish-url https://test.pypi.org/legacy/"
	uv publish --publish-url https://test.pypi.org/legacy/ dist/*

client-publish: client-clean ## Build and publish reflexio-client to PyPI
	@echo "→ uv build $(CLIENT_DIR)"
	cd $(CLIENT_DIR) && uv build
	@$(MAKE) verify-client-dist
	@echo "→ uv publish $(CLIENT_DIR)/dist/*"
	cd $(CLIENT_DIR) && uv publish dist/*

client-publish-dry: client-clean ## Build reflexio-client only; inspect dist/ without uploading
	@echo "→ uv build $(CLIENT_DIR) (dry: inspect $(CLIENT_DIR)/dist/ manually)"
	cd $(CLIENT_DIR) && uv build
	@ls -la $(CLIENT_DIR)/dist/

client-test-pypi: client-clean ## Build and publish reflexio-client to TestPyPI
	@echo "→ uv build $(CLIENT_DIR)"
	cd $(CLIENT_DIR) && uv build
	@echo "→ uv publish --publish-url https://test.pypi.org/legacy/"
	cd $(CLIENT_DIR) && uv publish --publish-url https://test.pypi.org/legacy/ dist/*

release: check-version check-clean check-branch check-up-to-date check-tag-free bump ## Bump + commit + tag + publish + push
	@echo "→ committing release v$(VERSION)"
	git add $(PYPROJECT) uv.lock
	git commit -m "Release v$(VERSION)"
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@$(MAKE) publish
	@echo "→ pushing commit + tag"
	git push --follow-tags
	@echo "✓ released reflexio-ai v$(VERSION) to PyPI"

client-release: check-client-version check-clean check-branch check-up-to-date check-client-tag-free client-bump client-test ## Bump, test, commit, tag, publish, and push reflexio-client only
	@echo "→ committing reflexio-client release v$(VERSION)"
	git add $(CLIENT_PYPROJECT) $(CLIENT_DIR)/uv.lock
	git commit -m "Release reflexio-client v$(VERSION)"
	git tag -a reflexio-client-v$(VERSION) -m "Release reflexio-client v$(VERSION)"
	@$(MAKE) client-publish
	@echo "→ pushing commit + tag"
	git push --follow-tags
	@echo "✓ released reflexio-client v$(VERSION) to PyPI"

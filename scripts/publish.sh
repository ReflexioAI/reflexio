#!/usr/bin/env bash
# Publish the reflexio package to PyPI.
#
# Usage:
#   ./scripts/publish.sh              # Build and upload to PyPI
#   ./scripts/publish.sh --dry-run    # Build only, validate, don't upload
#   ./scripts/publish.sh --test-pypi  # Upload to TestPyPI instead
#
# Prerequisites:
#   uv sync (installs build and twine from dev dependencies)
#
# For PyPI uploads, set TWINE_USERNAME and TWINE_PASSWORD (or use __token__ + API token).
# For TestPyPI, set TWINE_USERNAME and TWINE_PASSWORD for test.pypi.org.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=false
TEST_PYPI=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        --test-pypi) TEST_PYPI=true ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--test-pypi]"
            echo ""
            echo "  --dry-run    Build and validate only, don't upload"
            echo "  --test-pypi  Upload to TestPyPI instead of PyPI"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo "==> Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

echo "==> Building sdist and wheel..."
uv run python -m build

echo "==> Validating with twine..."
uv run twine check dist/*

# Show what was built
echo ""
echo "==> Built artifacts:"
ls -lh dist/

if $DRY_RUN; then
    echo ""
    echo "==> Dry run complete. Artifacts in dist/ are ready for upload."
    echo "    To upload to PyPI:     uv run twine upload dist/*"
    echo "    To upload to TestPyPI: uv run twine upload --repository testpypi dist/*"
    exit 0
fi

if $TEST_PYPI; then
    echo ""
    echo "==> Uploading to TestPyPI..."
    uv run twine upload --repository testpypi dist/*
    echo ""
    echo "==> Published to TestPyPI. Install with:"
    echo "    pip install --index-url https://test.pypi.org/simple/ reflexio"
else
    echo ""
    echo "==> Uploading to PyPI..."
    uv run twine upload dist/*
    echo ""
    echo "==> Published to PyPI. Install with:"
    echo "    pip install reflexio-ai"
fi

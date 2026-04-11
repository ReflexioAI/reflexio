# Reflexio

See [developer.md](developer.md) for full development guidelines, project structure, and setup instructions.

## Quick Reference

- Start services: `./run_services.sh` — runs FastAPI backend (port 8081) and Next.js docs (port 3000)
- Run commands in uv env: `uv run <cmd>` or activate `.venv`
- Use `curl` for API testing (faster); use Chrome for frontend tasks
- Never change env variable values in `.env` file — use shell exports for overrides
- API schemas live in `reflexio/models/api_schema/`

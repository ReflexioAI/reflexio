# Reflexio API Documentation Site

Interactive API reference and documentation for the Reflexio platform. Built with Next.js.

**Hosted docs:** https://www.reflexio.ai/docs

## What's Inside

- API reference for all REST endpoints (interactions, profiles, playbooks, config, search)
- Interactive API explorer for testing requests against a running server
- Schema documentation generated from the Reflexio backend

## Development Setup

```bash
cd docs
npm install
npm run dev
```

The site runs on **port 3000** by default. When started via `run_services.sh` from the project root, it runs on port 8062 instead.

## Build

```bash
npm run build
```

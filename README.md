# Time-series Foundational Models Library Monorepo

Contains repos for the projects. Each repo is a seperate folder.

## Setup for development

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. ```bash
uv sync
```

To add any dependency: `uv add <dependency>` or `uv add <dependency> --dev`

## Running programs

You can run programs as `uv run python ....` or `uv run ipython` where the prefix `uv run` is required to run the program in the virtual environment.
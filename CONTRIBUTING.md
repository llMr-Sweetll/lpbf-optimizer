# Contributing to LPBF-Optimizer

Thank you for your interest in improving LPBF-Optimizer!

## Development Setup

```bash
git clone https://github.com/llMr-Sweetll/lpbf-optimizer.git
cd lpbf-optimizer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest
```

## Code Style

This project uses `ruff` for linting and formatting:

```bash
ruff check src tests
ruff format src tests
```

Install pre-commit hooks:

```bash
pre-commit install
```

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Make your changes, adding tests where appropriate.
3. Ensure `pytest` passes locally.
4. Update documentation if your change affects usage.
5. Open a pull request using the provided template.

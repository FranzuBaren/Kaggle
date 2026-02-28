# Contributing to CXR-Sentinel

Thank you for your interest in contributing to CXR-Sentinel. This project aims to improve the safety and reliability of clinical AI systems, and contributions are welcome.

## Development Setup

```bash
git clone https://github.com/francescoorsi/cxr-sentinel.git
cd cxr-sentinel
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
# All tests (CPU-only, no GPU required)
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Specific test file
pytest tests/test_metrics.py -v
```

## Code Quality

```bash
# Lint
ruff check src/ tests/ scripts/

# Format
ruff format src/ tests/ scripts/

# Type check
mypy src/ --ignore-missing-imports
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for new functionality
4. Ensure all tests pass and lint is clean
5. Submit a pull request with a clear description

## Clinical Safety Note

CXR-Sentinel is designed for clinical environments. When contributing:

- Never weaken safety thresholds without clinical justification
- Always maintain the separation between generation (Diagnostician) and validation (FactChecker)
- New features should include appropriate test coverage
- Document any changes to the validation methodology

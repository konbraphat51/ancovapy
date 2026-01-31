# Contributing to ancovapy

Thank you for your interest in contributing to ancovapy! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/konbraphat51/ancovapy.git
cd ancovapy
```

2. Install uv (if not already installed):
```bash
pip install uv
```

3. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Development Workflow

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

Before submitting a PR, run:

```bash
# Format code
black ancovapy/ tests/ examples/

# Check linting
ruff check ancovapy/ tests/ examples/

# Type checking
mypy ancovapy/
```

### Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=ancovapy --cov-report=html
```

### Adding New Features

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Implement your feature with:
   - Proper type hints
   - Docstrings in Google style
   - Unit tests
   - Example usage (if applicable)

3. Ensure all tests pass and code quality checks succeed

4. Update documentation as needed

5. Submit a pull request with a clear description of changes

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_<functionality>_<scenario>`
- Include tests for:
  - Normal operation
  - Edge cases
  - Error conditions
  - Input validation

Example:
```python
def test_ancova_basic_functionality() -> None:
    """Test basic ANCOVA with simple dataset."""
    # Setup
    data = create_test_data()
    
    # Execute
    result = ancova.fit(data)
    
    # Assert
    assert result.p_value < 0.05
```

### Documentation

- All public functions must have docstrings
- Use Google-style docstrings
- Include parameter types and descriptions
- Provide usage examples for complex functions
- Update README.md for major features

### Commit Messages

Use clear, descriptive commit messages:

```
Add support for weighted ANCOVA

- Implement weight parameter in ANCOVA.fit()
- Add tests for weighted analysis
- Update documentation with examples
```

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about usage or development

## License

By contributing to ancovapy, you agree that your contributions will be licensed under the MIT License.

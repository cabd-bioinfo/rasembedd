# Test Suite for Protein Embedding Project

This directory contains a comprehensive test suite for the protein embedding and visualization project. The tests cover all major components including embedding generation, visualization, interactive dashboards, and the complete pipeline.

## 📁 Test Structure

```
tests/
├── __init__.py                          # Test package initialization
├── conftest.py                          # Shared fixtures and utilities
├── test_base_model.py                   # Tests for base model functionality
├── test_embedding_generation.py         # Tests for embedding generation
├── test_visualizations.py              # Tests for visualization functions
├── test_interactive_visualizations.py  # Tests for interactive Dash app
├── test_models.py                       # Tests for individual model implementations
└── test_integration.py                  # Integration and end-to-end tests
```

## 🧪 Test Categories

### Unit Tests
- **Base Model Tests** (`test_base_model.py`): Tests for the abstract base class and common functionality
- **Embedding Generation Tests** (`test_embedding_generation.py`): Tests for FASTA loading, embedding generation, and file I/O
- **Visualization Tests** (`test_visualizations.py`): Tests for distance metrics, projections, plotting functions
- **Interactive Tests** (`test_interactive_visualizations.py`): Tests for Dash app components and callbacks
- **Model Tests** (`test_models.py`): Tests for individual model implementations (ProstT5, ESM, etc.)

### Integration Tests
- **Pipeline Tests** (`test_integration.py`): End-to-end tests covering the complete workflow
- **Scalability Tests**: Tests with larger datasets and performance considerations
- **Cross-platform Tests**: Tests for compatibility across different environments

## 🚀 Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements_test.txt
```

### Quick Start

Run all tests:
```bash
python run_tests.py
```

### Test Runner Options

The `run_tests.py` script provides many options for running tests:

```bash
# Run with coverage reporting
python run_tests.py --coverage

# Run tests in parallel
python run_tests.py --parallel

# Run only fast tests (skip integration)
python run_tests.py --fast

# Run only specific test categories
python run_tests.py --models
python run_tests.py --visualizations
python run_tests.py --interactive
python run_tests.py --integration

# Run with HTML coverage report
python run_tests.py --coverage --html-report

# Run specific test files
python run_tests.py tests/test_models.py

# Verbose output
python run_tests.py --verbose
```

### Direct pytest Usage

You can also run tests directly with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=models --cov=generate_embeddings --cov-report=html

# Run specific test file
pytest tests/test_visualizations.py

# Run tests matching a pattern
pytest -k "test_distance"

# Run tests with markers
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
```

## 📊 Test Coverage

The test suite aims for high coverage across all components:

- **Models Package**: Base classes, individual model implementations
- **Embedding Generation**: FASTA parsing, model loading, embedding computation
- **Visualizations**: Distance calculations, projections, plotting
- **Interactive Components**: Dash app, callbacks, file uploads
- **Integration**: Complete pipeline workflows

### Coverage Reports

Generate coverage reports:
```bash
# Terminal report
python run_tests.py --coverage

# HTML report (opens in browser)
python run_tests.py --coverage --html-report
open htmlcov/index.html
```

## 🏷️ Test Markers

Tests are marked with categories for selective execution:

- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.models`: Model-related tests
- `@pytest.mark.visualizations`: Visualization tests
- `@pytest.mark.interactive`: Interactive app tests
- `@pytest.mark.gpu`: GPU-dependent tests
- `@pytest.mark.network`: Network-dependent tests

### Using Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Run only model tests
pytest -m "models"

# Run integration tests only
pytest -m "integration"
```

## 🔧 Test Configuration

### pytest.ini

The `pytest.ini` file contains test configuration:
- Test discovery patterns
- Output options
- Timeout settings
- Warning filters
- Logging configuration

### Fixtures

Shared test fixtures are defined in `conftest.py`:
- `sample_sequences`: Protein sequences for testing
- `sample_metadata`: Metadata DataFrame
- `sample_embeddings`: Mock embeddings
- `temp_fasta_file`: Temporary FASTA file
- `temp_metadata_file`: Temporary metadata file
- `temp_embeddings_file`: Temporary embeddings file
- `temp_dir`: Temporary directory

## 📋 Test Data

### Sample Data

The test suite uses realistic but small sample data:
- **3 protein sequences** of varying lengths
- **Metadata** with family, species, and length information
- **Mock embeddings** with realistic dimensions (128-1024D)

### Temporary Files

Tests automatically create and clean up temporary files:
- FASTA files for sequence loading tests
- Pickle files for embedding I/O tests
- TSV/CSV files for metadata tests
- Output directories for visualization tests

## 🚨 Common Issues and Solutions

### Import Errors

If you encounter import errors:
```bash
# Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/rasembedd"

# Or run from project root
cd /path/to/rasembedd
python -m pytest tests/
```

### Missing Dependencies

Install optional dependencies for full test coverage:
```bash
# For PaCMAP tests
pip install pacmap

# For HDF5 tests  
pip install h5py

# For interactive tests
pip install dash plotly

# For all model tests
pip install torch transformers
```

### GPU Tests

GPU-dependent tests are automatically skipped if CUDA is not available:
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Slow Tests

Some tests may be slow due to:
- Model loading (mocked in most tests)
- Large matrix operations
- File I/O operations

Use the `--fast` flag to skip integration tests:
```bash
python run_tests.py --fast
```

## 🔍 Debugging Tests

### Verbose Output

Use verbose mode for detailed test output:
```bash
python run_tests.py --verbose
```

### Running Single Tests

Run individual test functions:
```bash
pytest tests/test_models.py::TestProstT5Model::test_prost_t5_initialization -v
```

### Test Debugging

Add debugging to tests:
```python
import pytest

def test_my_function():
    result = my_function()
    pytest.set_trace()  # Debugger breakpoint
    assert result == expected
```

## 📈 Performance Testing

### Benchmark Tests

Run performance benchmarks:
```bash
python run_tests.py --benchmark
```

### Memory Testing

Monitor memory usage during tests:
```bash
# Install memory profiler
pip install memory-profiler

# Run with memory monitoring
python -m memory_profiler run_tests.py
```

## 🤝 Contributing Tests

### Adding New Tests

1. **Choose the appropriate test file** based on functionality
2. **Use existing fixtures** from `conftest.py` when possible
3. **Follow naming conventions**: `test_*` for functions, `Test*` for classes
4. **Add appropriate markers** for test categorization
5. **Include docstrings** describing what the test validates

### Test Guidelines

- **Test one thing per test function**
- **Use descriptive test names**
- **Mock external dependencies** (models, network calls)
- **Clean up resources** (use fixtures for temporary files)
- **Test both success and failure cases**
- **Include edge cases** (empty inputs, invalid data)

### Example Test Structure

```python
def test_function_name(fixture1, fixture2):
    """Test description of what is being validated."""
    # Arrange
    input_data = prepare_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_result
    assert_helper_function(result)
```

## 📝 Test Reports

### JUnit XML

Generate JUnit XML reports for CI/CD:
```bash
python run_tests.py --junit-xml=test-results.xml
```

### Coverage Reports

Multiple coverage report formats:
```bash
# Terminal report
python run_tests.py --coverage

# HTML report
python run_tests.py --coverage --html-report

# XML report (for CI/CD)
pytest --cov=models --cov-report=xml
```

## 🔄 Continuous Integration

The test suite is designed to work with CI/CD systems:

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - run: pip install -r requirements_test.txt
    - run: python run_tests.py --coverage --junit-xml=test-results.xml
    - uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: test-results.xml
```

## 📚 Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Python testing best practices](https://docs.python-guide.org/writing/tests/)
- [Mock object library](https://docs.python.org/3/library/unittest.mock.html)

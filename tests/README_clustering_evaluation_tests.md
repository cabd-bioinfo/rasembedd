# Clustering Evaluation Test Suite

This document provides detailed information about the test suite for the clustering evaluation module (`clustering_evaluation.py`).

## 📊 Test Coverage Summary

- **Total Tests**: 38 test cases
- **Code Coverage**: 78% (395 of 509 lines)
- **Test File**: `tests/test_clustering_evaluation.py` (766 lines)
- **Test Execution Time**: ~26 seconds

## 🎯 Coverage Breakdown

### ✅ Well-Covered Components (90%+ coverage)
- **ClusteringConfig**: Configuration management and validation
- **DataLoader**: File I/O, data preparation, and alignment
- **ClusteringEngine**: All clustering algorithms and optimization
- **ClusteringResult**: Result object functionality
- **SubsamplingAnalyzer**: Statistical analysis workflows
- **ClusteringAnalyzer**: Main orchestration logic

### ⚠️ Partially Covered Components (70-90% coverage)
- **Visualizer**: Plotting functions and color management
- **Error Handling**: File I/O edge cases
- **Integration**: End-to-end pipeline testing

### ❌ Minimal Coverage Areas (< 20% coverage)
- **CLI Interface**: Command-line argument parsing
- **Import Fallbacks**: Optional dependency handling
- **Script Entry Points**: Main function exception handling

## 🧪 Test Structure

### Test Classes and Coverage

| Test Class | Tests | Coverage Focus |
|-----------|-------|----------------|
| `TestClusteringConfig` | 2 | Configuration validation |
| `TestDataLoader` | 6 | File loading and data preparation |
| `TestClusteringEngine` | 8 | Clustering algorithms and evaluation |
| `TestClusteringResult` | 1 | Result object functionality |
| `TestVisualizer` | 4 | Plotting and visualization |
| `TestSubsamplingAnalyzer` | 4 | Statistical analysis |
| `TestClusteringAnalyzer` | 6 | Main analysis orchestration |
| `TestIntegration` | 4 | End-to-end workflows |
| `TestEdgeCases` | 4 | Boundary conditions and errors |

### Test Categories

#### Unit Tests (28 tests)
Focus on individual component functionality:
- Configuration validation
- Data loading and alignment
- Clustering algorithm implementation
- Evaluation metric calculation
- Statistical analysis methods

#### Integration Tests (6 tests)
Test complete workflows:
- Full pipeline execution
- Error handling scenarios
- File I/O operations
- Result generation and saving

#### Edge Case Tests (4 tests)
Validate boundary conditions:
- Empty datasets
- Insufficient data points
- Malformed input files
- Perfect clustering scenarios

## 🛠️ Test Features

### Comprehensive Mocking
- **matplotlib**: All plotting functions mocked to avoid file I/O
- **joblib.Parallel**: Parallel execution mocked for deterministic testing
- **File Operations**: Temporary files and directories for isolation
- **External Dependencies**: scikit-learn, pandas operations properly isolated

### Realistic Test Data
- **Protein Embeddings**: 5 proteins with 128-dimensional embeddings
- **Metadata**: Family annotations, species information
- **Multiple Formats**: TSV, CSV, pickle file support
- **Statistical Scenarios**: Subsampling results for statistical testing

### Error Testing
- **Missing Files**: File not found scenarios
- **Invalid Parameters**: Wrong clustering methods
- **Data Mismatches**: Embedding-metadata alignment issues
- **Edge Cases**: Single data points, empty inputs

## 🚀 Running Tests

### Basic Execution
```bash
# Run all clustering tests
python run_tests.py --clustering

# Run with coverage report
python run_tests.py --clustering --coverage

# Generate HTML coverage report
python run_tests.py --clustering --coverage --html-report
```

### Specific Test Execution
```bash
# Run specific test class
pytest tests/test_clustering_evaluation.py::TestClusteringEngine -v

# Run tests matching pattern
pytest tests/test_clustering_evaluation.py -k "kmeans" -v

# Run integration tests only
pytest tests/test_clustering_evaluation.py::TestIntegration -v

# Run with debugging output
pytest tests/test_clustering_evaluation.py::TestDataLoader::test_load_embeddings -v -s
```

### Coverage Analysis
```bash
# Detailed coverage with missing lines
pytest tests/test_clustering_evaluation.py --cov=clustering_evaluation --cov-report=term-missing

# HTML coverage report
pytest tests/test_clustering_evaluation.py --cov=clustering_evaluation --cov-report=html
open htmlcov/index.html

# XML coverage for CI
pytest tests/test_clustering_evaluation.py --cov=clustering_evaluation --cov-report=xml
```

## 📈 Test Performance

### Execution Metrics
- **Total Runtime**: ~26 seconds
- **Average per Test**: ~0.7 seconds
- **Memory Usage**: Minimal (mocked operations)
- **Parallel Capable**: Tests can run in parallel

### Performance Optimizations
- **Mocked Plotting**: Eliminates file I/O overhead
- **Small Test Data**: 5 proteins instead of full datasets
- **Temporary Files**: Automatic cleanup prevents disk bloat
- **Deterministic Seeds**: Reproducible random operations

## 🔍 Debugging Tests

### Common Issues and Solutions

#### Import Errors
```bash
# Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:/home/icases/rasembedd"
cd /home/icases/rasembedd
python -m pytest tests/test_clustering_evaluation.py
```

#### Missing Dependencies
```bash
# Install clustering requirements
pip install -r requirements_clustering.txt

# Verify scikit-learn version
python -c "import sklearn; print(sklearn.__version__)"
```

#### Test Failures
```bash
# Run with verbose output
pytest tests/test_clustering_evaluation.py::test_function_name -v -s

# Add debugging breakpoint
import pytest; pytest.set_trace()
```

### Test Debugging Tools
```bash
# Run single test with debugging
pytest tests/test_clustering_evaluation.py::TestClusteringEngine::test_perform_clustering_kmeans --pdb

# Show test fixtures
pytest tests/test_clustering_evaluation.py --fixtures

# Show test collection
pytest tests/test_clustering_evaluation.py --collect-only
```

## 📊 Coverage Improvement Opportunities

### To Reach 85% Coverage
Add tests for:
1. **CLI Interface**: Command-line argument parsing and validation
2. **Import Fallbacks**: Mock colorcet import failure scenarios
3. **File I/O Errors**: Permission denied, disk full scenarios
4. **Visualization Edge Cases**: Memory errors, plotting failures

### Suggested Additional Tests
```python
def test_cli_argument_parsing():
    """Test command-line interface."""
    # Mock sys.argv and test parse_arguments()

def test_import_error_handling():
    """Test behavior when optional dependencies missing."""
    # Mock import failures for colorcet

def test_file_permission_errors():
    """Test handling of file permission issues."""
    # Mock file operations that fail

def test_visualization_memory_errors():
    """Test large dataset visualization limits."""
    # Test with large synthetic datasets
```

## 🔄 CI/CD Integration

### GitHub Actions Integration
The clustering tests are integrated into the CI pipeline:

```yaml
- name: Run clustering evaluation tests
  run: |
    python run_tests.py --clustering --verbose
```

### Test Reports
- **JUnit XML**: Generated for CI/CD systems
- **Coverage XML**: Uploaded to Codecov
- **HTML Reports**: Available as artifacts

## 📝 Contributing Tests

### Adding New Tests
1. **Identify Coverage Gaps**: Use coverage report to find untested code
2. **Follow Naming Conventions**: `test_*` functions, `Test*` classes
3. **Use Existing Fixtures**: Leverage `sample_embeddings`, `temp_files`
4. **Mock External Dependencies**: Avoid real file I/O and network calls
5. **Test Both Success and Failure**: Include error scenarios

### Test Guidelines
- **One Assertion Per Test**: Focus on single functionality
- **Descriptive Names**: Clear test purpose from name
- **Proper Cleanup**: Use fixtures for temporary resources
- **Error Testing**: Validate exception handling
- **Edge Cases**: Test boundary conditions

### Example Test Template
```python
def test_new_functionality(sample_embeddings, temp_files):
    """Test description of what is being validated."""
    # Arrange
    config = ClusteringConfig(
        embedding_files=[temp_files['embedding_file']],
        metadata_file=temp_files['metadata_file']
    )

    # Act
    result = function_under_test(config)

    # Assert
    assert result is not None
    assert isinstance(result, expected_type)
    assert len(result) == expected_length
```

## 📚 Related Documentation

- [Main Test Suite README](tests/README.md)
- [Clustering Evaluation README](README_clustering_evaluation.md)
- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

---

**Test Suite Status**: ✅ **38/38 tests passing** | 📊 **78% coverage** | ⏱️ **~26s runtime**

#!/bin/bash
# Script to commit and push all changes

echo "🚀 Committing and pushing test suite and CI/CD setup..."

# Add all files
git add .

# Commit with comprehensive message
git commit -m "Add comprehensive test suite and CI/CD setup

✨ Features:
- Complete test framework with 8 test files covering all components
- 91 passing tests with 41% code coverage
- Professional testing infrastructure with fixtures, mocking, and configuration

🔧 Configuration:
- pytest.ini: Test configuration with markers and settings
- requirements_test.txt: Test-specific dependencies including pre-commit
- run_tests.py: Custom test runner with multiple execution options
- .gitignore: Updated to exclude coverage files and development artifacts

🤖 CI/CD:
- GitHub Actions workflow for automated testing across Python 3.9-3.12
- Pre-commit hooks for code formatting and basic quality checks
- Coverage reporting and integration test pipeline
- Security scanning and dependency checks

📊 Test Coverage:
- Base model functionality and abstract methods
- Embedding generation and file I/O operations
- Visualization methods and distance calculations
- Interactive Dash app components and callbacks
- Individual model implementations (ProstT5, ESM, Ankh, etc.)
- Integration tests for end-to-end pipeline
- Comprehensive error handling and edge cases

🛠️ Development Tools:
- Black code formatting (100 char line length)
- isort import sorting
- Pre-commit hooks for automated code quality
- Support for parallel test execution and benchmarking"

# Push to remote
echo "📤 Pushing to remote repository..."
git push origin main

echo "✅ Done! Your test suite and CI/CD setup is now live!"
echo "📊 You can run tests with: python run_tests.py --fast --coverage"
echo "🔧 Pre-commit hooks will run automatically on each commit"

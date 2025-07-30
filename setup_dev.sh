#!/bin/bash
# Setup script for development environment

echo "🚀 Setting up development environment..."

# Install test requirements (includes pre-commit)
echo "📦 Installing test requirements..."
pip install -r requirements_test.txt

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Run pre-commit on all files to ensure everything is set up correctly
echo "🔍 Running pre-commit on all files..."
pre-commit run --all-files

echo "✅ Development environment setup complete!"
echo ""
echo "📋 Quick commands:"
echo "  Run tests:           python run_tests.py --fast"
echo "  Run with coverage:   python run_tests.py --coverage --fast"
echo "  Run pre-commit:      pre-commit run --all-files"
echo "  Update hooks:        pre-commit autoupdate"

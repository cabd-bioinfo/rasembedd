#!/bin/bash
# Setup script for development environment

echo "🚀 Setting up development environment..."

# Check if we have basic requirements
echo "🔍 Checking requirements..."
if command -v python &> /dev/null; then
    python check_requirements.py
    if [ $? -ne 0 ]; then
        echo "⚠️  Missing some requirements. Installing them now..."
        pip install -r requirements.txt
        pip install -r requirements_clustering.txt
    fi
else
    echo "❌ Python not found. Please install Python 3.7+ first."
    exit 1
fi

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

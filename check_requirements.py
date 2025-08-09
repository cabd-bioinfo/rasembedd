#!/usr/bin/env python3
"""
Requirements checker for the protein embedding toolkit.
Run this script to verify all dependencies are correctly installed.
"""

import importlib
import sys
from typing import List, Tuple


def check_module(
    module_name: str, version_attr: str = "__version__", min_version: str = None
) -> Tuple[bool, str]:
    """Check if a module can be imported and optionally verify minimum version."""
    try:
        module = importlib.import_module(module_name)
        if min_version and hasattr(module, version_attr):
            version = getattr(module, version_attr)
            return True, f"✓ {module_name} ({version})"
        else:
            return True, f"✓ {module_name}"
    except ImportError as e:
        return False, f"✗ {module_name}: {str(e)}"


def check_requirements():
    """Check all critical requirements for the toolkit."""

    print("Checking requirements for Protein Embedding Toolkit...")
    print("=" * 60)

    # Core requirements (should be in requirements.txt)
    core_modules = [
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
        "scipy",
        "torch",
        "transformers",
        "plotly",
        "dash",
        "seaborn",
        "statsmodels",
        "umap",
        "pacmap",
        "colorcet",
    ]

    # Clustering-specific requirements
    clustering_modules = ["hdbscan", "joblib"]

    # Interactive visualization requirements
    interactive_modules = ["gunicorn"]

    # Optional modules
    optional_modules = ["h5py", "sentencepiece"]

    all_good = True

    print("\n📦 Core Requirements:")
    for module in core_modules:
        success, message = check_module(module)
        print(f"  {message}")
        if not success:
            all_good = False

    print("\n🔬 Clustering Requirements:")
    for module in clustering_modules:
        success, message = check_module(module)
        print(f"  {message}")
        if not success:
            all_good = False

    print("\n🌐 Interactive Visualization Requirements:")
    for module in interactive_modules:
        success, message = check_module(module)
        print(f"  {message}")
        if not success:
            print("    Note: Only needed for interactive visualization features")

    print("\n🔧 Optional Requirements:")
    for module in optional_modules:
        success, message = check_module(module)
        print(f"  {message}")
        if not success:
            print(f"    Note: {module} is optional - some features may be limited")

    print("\n" + "=" * 60)

    # Test key functionality
    print("\n🧪 Testing Key Functionality:")

    # Test clustering evaluation
    try:
        from clustering_evaluation import ClusteringAnalyzer, parse_arguments

        print("  ✓ Clustering evaluation module loads successfully")
    except Exception as e:
        print(f"  ✗ Clustering evaluation failed: {e}")
        all_good = False

    # Test basic sklearn clustering
    try:
        from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering

        print("  ✓ All sklearn clustering algorithms available")
    except Exception as e:
        print(f"  ✗ sklearn clustering failed: {e}")
        all_good = False

    print("\n" + "=" * 60)

    if all_good:
        print("\n🎉 All requirements satisfied! You're ready to use the toolkit.")
        print("\nNext steps:")
        print("  1. Generate embeddings: python generate_embeddings.py --help")
        print("  2. Create visualizations: python generate_visualizations.py --help")
        print("  3. Evaluate clustering: python clustering_evaluation.py --help")
        return True
    else:
        print("\n❌ Some requirements are missing. Please install them using:")
        print("  pip install -r requirements.txt")
        print("  pip install -r requirements_clustering.txt  # For clustering features")
        print("  pip install -r requirements_interactive.txt  # For interactive features")
        return False


if __name__ == "__main__":
    success = check_requirements()
    sys.exit(0 if success else 1)

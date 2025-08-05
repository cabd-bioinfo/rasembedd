#!/usr/bin/env python3
"""
Test runner script for the protein embedding project.

This script provides convenient commands for running tests with different configurations.
"""

import argparse
import os
import subprocess
import sys
import time


def run_command(cmd, description=""):
    """Run a command and return the result."""
    if description:
        print(f"\n{'=' * 60}")
        print(f"🧪 {description}")
        print(f"{'=' * 60}")

    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    if result.returncode == 0:
        print(f"✅ Success ({end_time - start_time:.2f}s)")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ Failed ({end_time - start_time:.2f}s)")
        if result.stderr:
            print(f"Error: {result.stderr}")
        if result.stdout:
            print(f"Output: {result.stdout}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run tests for the protein embedding project")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage reporting")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--fast", action="store_true", help="Run only fast tests (skip integration)"
    )
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--models", action="store_true", help="Run only model tests")
    parser.add_argument(
        "--visualizations", action="store_true", help="Run only visualization tests"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run only interactive visualization tests",
    )
    parser.add_argument(
        "--clustering", action="store_true", help="Run only clustering evaluation tests"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--html-report", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--junit-xml", help="Generate JUnit XML report")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    parser.add_argument("tests", nargs="*", help="Specific test files or patterns to run")

    args = parser.parse_args()

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # Add coverage
    if args.coverage:
        cmd.extend(
            [
                "--cov=models",
                "--cov=generate_embeddings",
                "--cov=generate_visualizations",
                "--cov=interactive_visualizations",
                "--cov=clustering_evaluation",
            ]
        )
        cmd.extend(["--cov-report=term-missing"])

        if args.html_report:
            cmd.extend(["--cov-report=html"])

        # Always generate XML for CI/CD
        cmd.extend(["--cov-report=xml"])

    # Add parallel execution
    if args.parallel:
        import multiprocessing

        num_processes = min(4, multiprocessing.cpu_count())  # Limit to 4 processes
        cmd.extend(["-n", str(num_processes)])

    # Add timeout
    cmd.extend(["--timeout", str(args.timeout)])

    # Add JUnit XML report
    if args.junit_xml:
        cmd.extend(["--junit-xml", args.junit_xml])

    # Determine which tests to run
    if args.tests:
        # Specific tests specified
        cmd.extend(args.tests)
    elif args.fast:
        # Fast tests only (exclude integration)
        cmd.extend(["tests/", "-k", "not test_integration"])
    elif args.integration:
        # Integration tests only
        cmd.extend(["tests/test_integration.py"])
    elif args.models:
        # Model tests only
        cmd.extend(
            [
                "tests/test_base_model.py",
                "tests/test_models.py",
                "tests/test_embedding_generation.py",
            ]
        )
    elif args.visualizations:
        # Visualization tests only
        cmd.extend(["tests/test_visualizations.py"])
    elif args.interactive:
        # Interactive visualization tests only
        cmd.extend(["tests/test_interactive_visualizations.py"])
    elif args.clustering:
        # Clustering evaluation tests only
        cmd.extend(["tests/test_clustering_evaluation.py"])
    elif args.benchmark:
        # Benchmark tests
        cmd.extend(["--benchmark-only"])
        cmd.extend(["tests/"])
    else:
        # All tests
        cmd.extend(["tests/"])

    # Run the tests
    print("🚀 Starting protein embedding project tests...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    result = run_command(cmd, "Running pytest")

    # Print summary
    if result.returncode == 0:
        print("\n🎉 All tests passed!")

        if args.coverage and args.html_report:
            print("\n📊 Coverage report generated in htmlcov/index.html")

    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

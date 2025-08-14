#!/usr/bin/env python3
"""
Comprehensive Test Runner for Context Reference Store

This script provides various options for running the test suite,
including quick tests, stress tests, and coverage analysis.
"""

import os
import sys
import subprocess
import argparse
import time
from typing import List, Optional


def run_command(cmd: List[str], capture_output: bool = False) -> tuple:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")

    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd)
            return result.returncode, "", ""
    except Exception as e:
        print(f"Error running command: {e}")
        return 1, "", str(e)


def check_dependencies() -> bool:
    """Check if required test dependencies are available."""
    required_packages = ["pytest", "pytest-cov"]
    optional_packages = [
        "langchain",
        "langgraph",
        "llama-index",
        "composio-core",
        "sentence-transformers",
        "tiktoken",
        "psutil",
        "pytest-benchmark",
    ]

    print("🔍 Checking dependencies...")

    missing_required = []
    missing_optional = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"{package}")
        except ImportError:
            missing_required.append(package)
            print(f"{package} (required)")

    for package in optional_packages:
        try:
            if package == "llama-index":
                __import__("llama_index")
            elif package == "composio-core":
                __import__("composio")
            elif package == "pytest-benchmark":
                # pytest-benchmark is a pytest plugin, check differently
                import pytest

                print(f"{package} (optional)")
            else:
                __import__(package.replace("-", "_"))
                print(f"{package} (optional)")
        except ImportError:
            missing_optional.append(package)
            print(f"{package} (optional)")

    if missing_required:
        print(f"\nMissing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False

    if missing_optional:
        print(f"\nMissing optional packages: {', '.join(missing_optional)}")
        print(
            "Some tests may be skipped. Install with: pip install "
            + " ".join(missing_optional)
        )

    return True


def run_quick_tests() -> int:
    """Run quick tests only."""
    print("\nRunning quick tests...")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--quick",
        "-v",
        "--tb=short",
        "-x", 
        "-m",
        "not slow",
    ]

    returncode, _, _ = run_command(cmd)
    return returncode


def run_full_tests() -> int:
    """Run the full test suite."""
    print("\nRunning full test suite...")
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]

    returncode, _, _ = run_command(cmd)
    return returncode


def run_stress_tests() -> int:
    """Run stress and performance tests."""
    print("\nRunning stress tests...")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_stress_comprehensive.py",
        "-v",
        "--tb=short",
        "-s", 
    ]

    returncode, _, _ = run_command(cmd)
    return returncode


def run_edge_case_tests() -> int:
    """Run edge case tests."""
    print("\nRunning edge case tests...")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_edge_cases.py",
        "-v",
        "--tb=short",
    ]

    returncode, _, _ = run_command(cmd)
    return returncode


def run_adapter_tests() -> int:
    """Run adapter tests."""
    print("\nRunning adapter tests...")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_adapters_comprehensive.py",
        "-v",
        "--tb=short",
    ]

    returncode, _, _ = run_command(cmd)
    return returncode


def run_integration_tests() -> int:
    """Run integration tests."""
    print("\nRunning integration tests...")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_integration_comprehensive.py",
        "-v",
        "--tb=short",
        "-s",
    ]

    returncode, _, _ = run_command(cmd)
    return returncode


def run_with_coverage() -> int:
    """Run tests with coverage analysis."""
    print("\nRunning tests with coverage analysis...")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--cov=context_store",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "-v",
    ]

    returncode, stdout, stderr = run_command(cmd, capture_output=True)

    if stdout:
        print(stdout)
    if stderr:
        print(stderr)

    if returncode == 0:
        print("\nCoverage report generated:")
        print("  - HTML: htmlcov/index.html")
        print("  - XML: coverage.xml")

    return returncode


def run_benchmarks() -> int:
    """Run performance benchmarks."""
    print("\nRunning performance benchmarks...")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_stress_comprehensive.py::TestPerformanceBenchmarks",
        "--benchmark-only",
        "--benchmark-sort=mean",
        "-v",
    ]

    returncode, _, _ = run_command(cmd)
    return returncode


def run_specific_test(test_pattern: str) -> int:
    """Run a specific test or test pattern."""
    print(f"\nRunning specific test: {test_pattern}")
    cmd = [sys.executable, "-m", "pytest", test_pattern, "-v", "--tb=short"]

    returncode, _, _ = run_command(cmd)
    return returncode


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Context Reference Store Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --quick           # Run quick tests only
  python run_tests.py --full            # Run full test suite
  python run_tests.py --stress          # Run stress tests
  python run_tests.py --coverage        # Run with coverage
  python run_tests.py --adapters        # Run adapter tests only
  python run_tests.py --integration     # Run integration tests
  python run_tests.py --benchmark       # Run performance benchmarks
  python run_tests.py --test test_basic # Run specific test
        """,
    )

    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--stress", action="store_true", help="Run stress tests")
    parser.add_argument("--edge-cases", action="store_true", help="Run edge case tests")
    parser.add_argument("--adapters", action="store_true", help="Run adapter tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage analysis"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks"
    )
    parser.add_argument("--test", type=str, help="Run specific test pattern")
    parser.add_argument(
        "--check-deps", action="store_true", help="Check dependencies only"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print("Context Reference Store Test Runner")
    print("=" * 50)

    # Check dependencies first
    if not check_dependencies():
        if not args.check_deps:
            print("\nCannot proceed without required dependencies.")
            return 1
        else:
            return 1

    if args.check_deps:
        print("\nAll required dependencies are available!")
        return 0

    # Set environment variables for testing
    os.environ["PYTEST_DEBUG"] = "true" if args.verbose else "false"

    start_time = time.time()
    returncode = 0

    try:
        if args.quick:
            returncode = run_quick_tests()
        elif args.stress:
            returncode = run_stress_tests()
        elif args.edge_cases:
            returncode = run_edge_case_tests()
        elif args.adapters:
            returncode = run_adapter_tests()
        elif args.integration:
            returncode = run_integration_tests()
        elif args.coverage:
            returncode = run_with_coverage()
        elif args.benchmark:
            returncode = run_benchmarks()
        elif args.test:
            returncode = run_specific_test(args.test)
        elif args.full:
            returncode = run_full_tests()
        else:
            # Default: run quick tests
            print("No specific test type specified, running quick tests...")
            returncode = run_quick_tests()

    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        returncode = 130

    except Exception as e:
        print(f"\nError running tests: {e}")
        returncode = 1

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n{'='*50}")
    if returncode == 0:
        print(f"Tests completed successfully in {duration:.2f}s")
    else:
        print(f"Tests failed with exit code {returncode} after {duration:.2f}s")

    return returncode


if __name__ == "__main__":
    sys.exit(main())

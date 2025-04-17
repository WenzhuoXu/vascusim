"""
Master test script for vascusim package.

This script runs all tests, checks code coverage, and optionally builds the package.
Run with --help to see all available options.
"""

import os
import sys
import subprocess
import argparse
import shutil
import platform
from pathlib import Path


def check_dependencies():
    """Check if required testing dependencies are installed."""
    required_packages = [
        "pytest",
        "black",
        "isort",
        "flake8",
        "mypy"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required testing dependencies: {', '.join(missing)}")
        print("Install them with: pip install " + " ".join(missing))
        return False
    
    return True


def check_nas_connectivity(host, port=5000):
    """Check if a NAS is reachable for testing."""
    import socket
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ NAS at {host}:{port} is reachable")
            return True
        else:
            print(f"⚠️  NAS at {host}:{port} is not reachable. Some tests will be skipped.")
            return False
    except Exception as e:
        print(f"⚠️  Error checking NAS connectivity: {e}")
        return False


def run_code_quality_checks(args):
    """Run code formatting and quality checks."""
    print("\n===== Running code quality checks =====")
    
    # Check code formatting with black
    print("\n--- Checking code formatting with black ---")
    result = subprocess.run(
        ["black", "--check", "vascusim", "tests"],
        capture_output=not args.verbose
    )
    if result.returncode != 0:
        if args.fix:
            print("Fixing code formatting with black...")
            subprocess.run(["black", "vascusim", "tests"])
        else:
            print("❌ Code formatting check failed. Run with --fix to auto-format.")
            if not args.ignore_errors:
                return False
    else:
        print("✅ Code formatting check passed")
    
    # Check import sorting with isort
    print("\n--- Checking import sorting with isort ---")
    result = subprocess.run(
        ["isort", "--check", "vascusim", "tests"],
        capture_output=not args.verbose
    )
    if result.returncode != 0:
        if args.fix:
            print("Fixing import sorting with isort...")
            subprocess.run(["isort", "vascusim", "tests"])
        else:
            print("❌ Import sorting check failed. Run with --fix to auto-format.")
            if not args.ignore_errors:
                return False
    else:
        print("✅ Import sorting check passed")
    
    # Check code style with flake8
    print("\n--- Checking code style with flake8 ---")
    result = subprocess.run(
        ["flake8", "vascusim", "tests"],
        capture_output=not args.verbose
    )
    if result.returncode != 0:
        print("❌ Code style check failed.")
        if not args.ignore_errors:
            return False
    else:
        print("✅ Code style check passed")
    
    # Run type checking with mypy if enabled
    if args.type_check:
        print("\n--- Running type checking with mypy ---")
        result = subprocess.run(
            ["mypy", "vascusim"],
            capture_output=not args.verbose
        )
        if result.returncode != 0:
            print("❌ Type checking failed.")
            if not args.ignore_errors:
                return False
        else:
            print("✅ Type checking passed")
    
    return True


def run_tests(args):
    """Run tests with pytest and collect coverage."""
    print("\n===== Running tests =====")
    
    # Build command
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add coverage if enabled
    if args.coverage:
        cmd.extend(["--cov=vascusim", "--cov-report=term"])
        if args.coverage_html:
            cmd.append("--cov-report=html")
    
    # Add specific tests if provided
    if args.test_path:
        cmd.append(args.test_path)
    else:
        cmd.append("tests/")
    
    # Run tests
    result = subprocess.run(cmd)
    
    # Check result
    if result.returncode != 0:
        print("❌ Tests failed")
        return False
    else:
        print("✅ All tests passed")
        return True


def build_package(args):
    """Build the package using the specified method."""
    print("\n===== Building package =====")
    
    if args.build_method == "setuptools":
        cmd = [sys.executable, "-m", "build"]
    elif args.build_method == "wheel":
        cmd = [sys.executable, "-m", "pip", "wheel", "-w", "dist", "."]
    else:
        cmd = [sys.executable, "setup.py", "sdist", "bdist_wheel"]
    
    # Run build command
    result = subprocess.run(cmd, capture_output=not args.verbose)
    
    if result.returncode != 0:
        print("❌ Package build failed")
        if args.verbose and result.stdout:
            print(result.stdout.decode())
        if result.stderr:
            print(result.stderr.decode())
        return False
    else:
        print("✅ Package built successfully")
        # List built packages
        if os.path.exists("dist"):
            print("\nBuilt packages:")
            for package in os.listdir("dist"):
                print(f"  - {package}")
        return True


def clean_build_artifacts():
    """Clean build artifacts."""
    print("\n===== Cleaning build artifacts =====")
    
    paths_to_remove = [
        "build",
        "dist",
        "*.egg-info",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "**/__pycache__"
    ]
    
    for path in paths_to_remove:
        for item in Path(".").glob(path):
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
    
    print("✅ Build artifacts cleaned")


def main():
    """Main function to run tests and build package."""
    parser = argparse.ArgumentParser(description="Run tests and build vascusim package")
    
    # Test options
    parser.add_argument("--test-path", help="Specific test module or directory to run")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--coverage-html", action="store_true", help="Generate HTML coverage report")
    
    # Code quality options
    parser.add_argument("--skip-quality", action="store_true", help="Skip code quality checks")
    parser.add_argument("--fix", action="store_true", help="Auto-fix code formatting issues")
    parser.add_argument("--type-check", action="store_true", help="Run type checking with mypy")
    
    # Build options
    parser.add_argument("--build", action="store_true", help="Build package after tests")
    parser.add_argument("--build-method", choices=["setuptools", "wheel", "legacy"], 
                      default="setuptools", help="Package build method")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts before running")
    
    # NAS testing options
    parser.add_argument("--nas-host", help="NAS hostname or IP for connection testing")
    parser.add_argument("--nas-port", type=int, default=5000, help="NAS port for connection testing")
    
    # General options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--ignore-errors", action="store_true", 
                      help="Continue even if some checks fail")
    
    args = parser.parse_args()
    
    # Show banner
    print("\n" + "=" * 60)
    print(f"  VASCUSIM TEST RUNNER - {platform.python_version()}")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        if not args.ignore_errors:
            sys.exit(1)
    
    # Clean if requested
    if args.clean:
        clean_build_artifacts()
    
    # Check NAS connectivity if host provided
    if args.nas_host:
        check_nas_connectivity(args.nas_host, args.nas_port)
    
    # Set initial result status
    success = True
    
    # Run code quality checks if not skipped
    if not args.skip_quality:
        if not run_code_quality_checks(args):
            success = False
            if not args.ignore_errors:
                sys.exit(1)
    
    # Run tests if not skipped
    if not args.skip_tests:
        if not run_tests(args):
            success = False
            if not args.ignore_errors:
                sys.exit(1)
    
    # Build package if requested and all tests passed
    if args.build and (success or args.ignore_errors):
        if not build_package(args):
            success = False
            if not args.ignore_errors:
                sys.exit(1)
    
    # Print final status
    if success:
        print("\n✅ All tasks completed successfully")
    else:
        print("\n⚠️  Some tasks failed. See output for details.")
        if not args.ignore_errors:
            sys.exit(1)


if __name__ == "__main__":
    main()
# Contributing to Context Reference Store

Thank you for your interest in contributing to Context Reference Store! This document provides guidelines and information for contributors.

## Table of Contents

- [Contributing to Context Reference Store](#contributing-to-context-reference-store)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Development Setup](#development-setup)
    - [Repository Structure](#repository-structure)
  - [Contributing Guidelines](#contributing-guidelines)
    - [Types of Contributions](#types-of-contributions)
    - [Before You Start](#before-you-start)
    - [Pull Request Process](#pull-request-process)
    - [Commit Message Guidelines](#commit-message-guidelines)
  - [Development Workflow](#development-workflow)
    - [Setting Up Your Environment](#setting-up-your-environment)
    - [Running Tests](#running-tests)
    - [Code Quality](#code-quality)
    - [Documentation](#documentation)
  - [Testing Guidelines](#testing-guidelines)
    - [Test Structure](#test-structure)
    - [Writing Tests](#writing-tests)
    - [Performance Testing](#performance-testing)
  - [Code Style and Standards](#code-style-and-standards)
    - [Python Code Style](#python-code-style)
    - [Type Hints](#type-hints)
    - [Documentation Standards](#documentation-standards)
  - [Reporting Issues](#reporting-issues)
    - [Bug Reports](#bug-reports)
    - [Feature Requests](#feature-requests)
    - [Performance Issues](#performance-issues)
  - [Development Tips](#development-tips)
    - [Debugging](#debugging)
    - [Performance Profiling](#performance-profiling)
    - [Memory Analysis](#memory-analysis)
  - [Release Process](#release-process)
  - [Community](#community)
    - [Communication](#communication)
    - [Getting Help](#getting-help)
  - [Recognition](#recognition)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code. Please report unacceptable behavior to [waleadenle1@gmail.com](mailto:waleadenle1@gmail.com).

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of AI agents and context management
- Familiarity with Python development best practices

### Development Setup

1. **Fork and Clone**

   ```bash
   # Fork the repository on GitHub
   git clone https://github.com/YOUR_USERNAME/Context_reference_store.git
   cd Context_reference_store
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify Installation**
   ```bash
   python -c "import context_store; print('Setup successful!')"
   ```

### Repository Structure

```
context-reference-store/
├── context_store/           # Main package
│   ├── core/               # Core functionality
│   ├── adapters/           # Framework adapters
│   ├── monitoring/         # Performance monitoring
│   ├── optimization/       # Performance optimization
│   ├── semantic/          # Semantic analysis
│   └── utils/             # Utility functions
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── performance/       # Performance tests
│   └── stress/            # Stress tests
├── docs/                  # Documentation
│   ├── tutorials/         # Tutorials and guides
│   ├── integrations/      # Framework integration guides
│   ├── api/              # API reference
│   └── guides/           # User guides
├── examples/             # Example applications
└── benchmarks/           # Performance benchmarks
```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

1. **Bug Fixes**

   - Fix existing bugs
   - Improve error handling
   - Enhance stability

2. **New Features**

   - Framework integrations
   - Performance optimizations
   - Monitoring capabilities
   - New cache policies

3. **Documentation**

   - API documentation
   - Tutorials and guides
   - Example applications
   - Integration guides

4. **Performance Improvements**

   - Algorithm optimizations
   - Memory efficiency
   - Compression improvements
   - Caching strategies

5. **Testing**
   - Unit tests
   - Integration tests
   - Performance benchmarks
   - Stress tests

### Before You Start

1. **Check Existing Issues**

   - Look for existing issues related to your contribution
   - Comment on issues to avoid duplicate work

2. **Discuss Major Changes**

   - Open an issue for discussion before starting major features
   - Get feedback on your approach

3. **Understand the Codebase**
   - Read the documentation
   - Run existing tests
   - Explore example applications

### Pull Request Process

1. **Create Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**

   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Test Your Changes**

   ```bash
   # Run all tests
   pytest

   # Run specific test categories
   pytest tests/unit/
   pytest tests/integration/
   pytest tests/performance/

   # Run with coverage
   pytest --cov=context_store
   ```

4. **Update Documentation**

   - Update relevant documentation
   - Add docstrings to new functions/classes
   - Update CHANGELOG.md

5. **Submit Pull Request**
   - Push your branch to your fork
   - Create pull request with clear description
   - Link related issues

### Commit Message Guidelines

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements

**Examples:**

```bash
feat(core): add TTL-based cache eviction policy

fix(adapters): resolve LangChain integration memory leak

docs(tutorials): add agent building tutorial

test(performance): add compression benchmark tests
```

## Development Workflow

### Setting Up Your Environment

1. **Install Pre-commit Hooks**

   ```bash
   pre-commit install
   ```

2. **Configure IDE**

   - Set up linting (flake8, black, isort)
   - Configure type checking (mypy)
   - Enable auto-formatting

3. **Environment Variables**
   ```bash
   # Optional: Set development environment
   export CONTEXT_STORE_ENV=development
   export CONTEXT_STORE_LOG_LEVEL=DEBUG
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/                    # Unit tests
pytest tests/integration/             # Integration tests
pytest tests/performance/             # Performance tests
pytest tests/stress/                  # Stress tests

# Run tests with coverage
pytest --cov=context_store --cov-report=html

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/unit/test_core.py

# Run specific test function
pytest tests/unit/test_core.py::test_store_and_retrieve
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code
flake8 context_store/
pylint context_store/

# Type checking
mypy context_store/

# Run all quality checks
./scripts/quality_check.sh
```

### Documentation

```bash
# Generate API documentation
sphinx-build -b html docs/ docs/_build/

# Serve documentation locally
python -m http.server 8000 --directory docs/_build/

# Check documentation links
sphinx-build -b linkcheck docs/ docs/_build/
```

## Testing Guidelines

### Test Structure

```python
# tests/unit/test_example.py
import pytest
from context_store import ContextReferenceStore
from context_store.exceptions import ContextNotFoundError

class TestContextReferenceStore:
    """Test suite for ContextReferenceStore"""

    def setup_method(self):
        """Setup before each test method"""
        self.store = ContextReferenceStore(cache_size=10)

    def teardown_method(self):
        """Cleanup after each test method"""
        self.store.clear()

    def test_store_and_retrieve(self):
        """Test basic store and retrieve functionality"""
        # Arrange
        data = {"test": "data"}

        # Act
        context_id = self.store.store(data)
        retrieved = self.store.retrieve(context_id)

        # Assert
        assert retrieved == data
        assert self.store.exists(context_id)

    def test_retrieve_nonexistent_context(self):
        """Test retrieving non-existent context raises error"""
        with pytest.raises(ContextNotFoundError):
            self.store.retrieve("nonexistent_id")
```

### Writing Tests

1. **Test Categories**

   - **Unit Tests**: Test individual components in isolation
   - **Integration Tests**: Test component interactions
   - **Performance Tests**: Measure performance characteristics
   - **Stress Tests**: Test under high load conditions

2. **Test Naming**

   ```python
   def test_method_name_when_condition_then_expected_result(self):
       """Test that method_name returns expected_result when condition"""
       pass
   ```

3. **Test Structure (AAA Pattern)**
   ```python
   def test_example(self):
       # Arrange - Set up test data and conditions
       store = ContextReferenceStore()
       data = {"test": "data"}

       # Act - Execute the functionality being tested
       context_id = store.store(data)

       # Assert - Verify the results
       assert store.exists(context_id)
   ```

### Performance Testing

```python
# tests/performance/test_benchmarks.py
import pytest
import time
from context_store import ContextReferenceStore

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    @pytest.mark.benchmark
    def test_store_performance(self, benchmark):
        """Benchmark store operation performance"""
        store = ContextReferenceStore()
        data = {"benchmark": "data"}

        # Benchmark the store operation
        result = benchmark(store.store, data)
        assert result is not None

    @pytest.mark.performance
    def test_large_context_handling(self):
        """Test handling of large contexts"""
        store = ContextReferenceStore(use_compression=True)

        # Create large context (1MB)
        large_data = {"data": "x" * (1024 * 1024)}

        start_time = time.time()
        context_id = store.store(large_data)
        store_time = time.time() - start_time

        start_time = time.time()
        retrieved = store.retrieve(context_id)
        retrieve_time = time.time() - start_time

        # Performance assertions
        assert store_time < 1.0  # Should store in under 1 second
        assert retrieve_time < 0.5  # Should retrieve in under 0.5 seconds
        assert retrieved == large_data
```

## Code Style and Standards

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Line length: 88 characters (Black default)
# Use double quotes for strings
# Use type hints for all public functions

class ExampleClass:
    """Example class demonstrating code style."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config
        self._private_attr = None

    def public_method(self, param: str, optional_param: int = 10) -> bool:
        """Public method with proper documentation.

        Args:
            param: Description of parameter
            optional_param: Optional parameter with default value

        Returns:
            Boolean indicating success

        Raises:
            ValueError: If param is invalid
        """
        if not param:
            raise ValueError("param cannot be empty")

        return True

    def _private_method(self) -> None:
        """Private method for internal use."""
        pass
```

### Type Hints

Use type hints for all public APIs:

```python
from typing import Dict, List, Optional, Union, Any, Callable
from typing import TypeVar, Generic, Protocol

# Generic types
T = TypeVar('T')

class Store(Generic[T]):
    """Generic store class."""

    def store(self, item: T) -> str:
        """Store item and return ID."""
        pass

    def retrieve(self, item_id: str) -> T:
        """Retrieve item by ID."""
        pass

# Union types for flexible APIs
def configure(self, policy: Union[str, CacheEvictionPolicy]) -> None:
    """Configure with string or enum."""
    pass

# Optional types
def get_metadata(self, context_id: str) -> Optional[Dict[str, Any]]:
    """Get metadata, may return None."""
    pass
```

### Documentation Standards

Use Google-style docstrings:

```python
def complex_function(
    param1: str,
    param2: List[int],
    param3: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """One-line summary of function purpose.

    Longer description providing more details about the function's
    behavior, algorithms used, or important considerations.

    Args:
        param1: Description of first parameter
        param2: List of integers for processing
        param3: Optional configuration dictionary

    Returns:
        Tuple containing:
            - Success boolean
            - Result message string

    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 contains non-integers

    Example:
        >>> success, message = complex_function("test", [1, 2, 3])
        >>> print(f"Success: {success}, Message: {message}")
        Success: True, Message: Processed successfully

    Note:
        This function has O(n) time complexity where n is len(param2).
    """
    pass
```

## Reporting Issues

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:

1. Create store with config...
2. Call method with parameters...
3. Observe error...

**Expected behavior**
What you expected to happen.

**Environment:**

- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- Context Store version: [e.g. 1.0.0]
- Framework versions: [e.g. LangChain 0.1.0]

**Additional context**
Any additional information about the problem.
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
Clear description of the feature you'd like to see.

**Use Case**
Describe the use case and why this feature would be valuable.

**Proposed Implementation**
If you have ideas about how to implement this feature.

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Any other context or screenshots about the feature request.
```

### Performance Issues

For performance-related issues:

```markdown
**Performance Issue Description**
Describe the performance problem you're experiencing.

**Performance Measurements**
Include specific measurements:

- Response times
- Memory usage
- CPU usage
- Throughput

**Configuration**
Include your store configuration:

- Cache size
- Compression settings
- Eviction policy
- Storage configuration

**Workload Characteristics**
Describe your workload:

- Context sizes
- Access patterns
- Concurrency level

**Environment Details**

- Hardware specifications
- Operating system
- Python version
- Other relevant software
```

## Development Tips

### Debugging

1. **Enable Debug Logging**

   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)

   # Or set environment variable
   export CONTEXT_STORE_LOG_LEVEL=DEBUG
   ```

2. **Use Debug Store Configuration**

   ```python
   debug_store = ContextReferenceStore(
       cache_size=10,  # Small cache for testing
       enable_metrics=True,  # Enable detailed metrics
       use_compression=False  # Disable compression for debugging
   )
   ```

3. **Inspect Internal State**

   ```python
   # Check cache state
   print(store._cache.keys())

   # Get detailed statistics
   stats = store.get_detailed_stats()
   print(json.dumps(stats, indent=2))
   ```

### Performance Profiling

1. **Using cProfile**

   ```bash
   python -m cProfile -s cumulative your_script.py
   ```

2. **Using py-spy**

   ```bash
   py-spy top --pid <python-process-id>
   py-spy record -o profile.svg --pid <python-process-id>
   ```

3. **Memory Profiling**

   ```python
   from memory_profiler import profile

   @profile
   def test_memory_usage():
       store = ContextReferenceStore()
       for i in range(1000):
           store.store(f"data_{i}")
   ```

### Memory Analysis

1. **Using tracemalloc**

   ```python
   import tracemalloc

   tracemalloc.start()

   # Your code here
   store = ContextReferenceStore()
   # ... operations ...

   current, peak = tracemalloc.get_traced_memory()
   print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
   print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

   tracemalloc.stop()
   ```

2. **Using objgraph**

   ```python
   import objgraph

   # Show most common objects
   objgraph.show_most_common_types()

   # Show growth in object counts
   objgraph.show_growth()
   ```

## Release Process

1. **Version Bumping**

   - Update version in `pyproject.toml`
   - Update version in `context_store/__init__.py`
   - Update CHANGELOG.md

2. **Testing**

   ```bash
   # Run full test suite
   pytest

   # Run performance benchmarks
   pytest tests/performance/

   # Test installation
   pip install -e .
   python -c "import context_store; print(context_store.__version__)"
   ```

3. **Documentation**

   - Update documentation
   - Generate API docs
   - Update README if needed

4. **Release**

   ```bash
   # Create release branch
   git checkout -b release/v1.x.x

   # Commit changes
   git add .
   git commit -m "chore: prepare release v1.x.x"

   # Create tag
   git tag v1.x.x

   # Push to repository
   git push origin release/v1.x.x
   git push origin v1.x.x
   ```

## Community

### Communication

- **GitHub Issues**: Bug reports, feature requests, discussions
- **GitHub Discussions**: General questions, ideas, community discussions
- **Email**: [waleadenle1@gmail.com](mailto:waleadenle1@gmail.com) for private matters

### Getting Help

1. **Documentation**: Check existing documentation first
2. **Search Issues**: Look for similar issues or discussions
3. **Ask Questions**: Open a discussion or issue if you can't find answers
4. **Join Community**: Participate in discussions and help others

## Recognition

Contributors are recognized in several ways:

1. **Contributors List**: All contributors are listed in the README
2. **Changelog**: Significant contributions are mentioned in release notes
3. **GitHub Recognition**: GitHub's built-in contributor recognition
4. **Special Recognition**: Outstanding contributors may receive special recognition

### Types of Recognition

- **Code Contributors**: Those who contribute code improvements
- **Documentation Contributors**: Those who improve documentation
- **Community Contributors**: Those who help with discussions and support
- **Testing Contributors**: Those who improve test coverage and quality

Thank you for contributing to Context Reference Store! Your contributions help make efficient context management accessible to the entire AI community.

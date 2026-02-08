# Contributing

Thank you for your interest in contributing to `MOSShell`! This document provides guidelines and instructions for contributing.

## Before You Start

We welcome contributions! These guidelines exist to save everyone time. Following them means your work is more likely to be accepted.

**All pull requests require a corresponding issue.** Unless your change is trivial (typo, docs tweak, broken link), create an issue first. Every merged feature becomes ongoing maintenance, so we need to agree something is worth doing before reviewing code. PRs without a linked issue will be closed.

## Development Setup

1. Make sure you have `Python 3.12+` installed
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
1. Fork the repository && Clone your fork
1. Install development dependencies: `make prepare`
1. Create a new branch && Make your changes
1. Run format, lint and tests before submitting a PR:

```bash
make format
make lint
make test
```

### Checklist

1. Update documentation as needed
1. Add tests for new functionality
1. Ensure CI passes
1. Address review feedback

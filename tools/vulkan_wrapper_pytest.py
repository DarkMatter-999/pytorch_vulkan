#!/usr/bin/env python3
"""Explicit wrapper-first pytest entry point; arguments and exit status are pytest's."""
import sys


def main():
    import pytorch_vulkan  # Register the backend before pytest import/collection.
    import pytest

    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

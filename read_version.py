import logging

try:  # catch for pytest
    import toml

    with open("pyproject.toml") as f:
        parsed_toml = toml.load(f)
    version = parsed_toml["tool"]["poetry"]["version"]

    print(version)
except ImportError:
    logging.error("toml not installed, cannot read version from pyproject.toml")

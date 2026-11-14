"""Allow ``python -m cervical_habitat ...``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

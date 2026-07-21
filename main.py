"""Run a lightweight model smoke test.

Use python -m stego.cli --help for training and evaluation commands.
"""

from stego.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["smoke"]))

"""Obtain the pinned official source into a new checkout; never replace a checkout."""
import argparse
from pathlib import Path
import subprocess


COMMIT = "aa017c34afbe023de9462b17ae6067b52fef9867"
URL = "https://github.com/tue-mps/eomt.git"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "upstream/eomt")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists; left unchanged: {destination}")
    subprocess.run(["git", "clone", "--no-checkout", URL, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", COMMIT], check=True)
    head = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise RuntimeError(f"Unexpected source revision: {head}")
    print(f"Official EoMT source: {head}")


if __name__ == "__main__":
    main()

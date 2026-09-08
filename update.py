import argparse
import json
import os
import re
import sys
import urllib.request
from urllib.error import URLError

REQUIREMENTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "requirements.txt"
)


def safe_input(prompt):
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)


def is_prerelease(version):
    return bool(re.search(r"[a-zA-Z]", version))


def version_key(version):
    parts = []
    for p in version.split("."):
        m = re.match(r"(\d+)", p)
        parts.append(int(m.group(1)) if m else 0)
    return parts


def latest_stable(package):
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.load(resp)
    except (URLError, ValueError):
        return None

    candidates = [v for v in data.get("releases", {}).keys() if not is_prerelease(v)]
    if not candidates:
        return None
    return max(candidates, key=version_key)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="update.py",
        description="Update pinned versions in requirements.txt to the latest stable PyPI releases.",
        epilog="Run with no flags for interactive one-by-one prompts, or pass --yes to update everything without asking.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="update all packages without prompting",
    )
    return parser.parse_args()


def update_requirements(yes):
    if not os.path.isfile(REQUIREMENTS_PATH):
        print(f"'requirements.txt' not found at {REQUIREMENTS_PATH}")
        sys.exit(1)

    with open(REQUIREMENTS_PATH, "r") as f:
        lines = f.readlines()

    print("Scanning dependencies...")

    changed = 0
    for i, line in enumerate(lines):
        m = re.match(r"\s*([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if not m:
            continue
        package, current = m.group(1), m.group(2)
        latest = latest_stable(package)
        if latest is None:
            print(f"Failed to fetch version for {package}, skipping...")
            continue
        if latest == current:
            continue
        if yes:
            print(f"Updating {package}: {current} -> {latest}")
        else:
            choice = safe_input(f"Update {package} from {current} to {latest} (Y/n)? ")
            if choice.lower() not in ["y", "yes", ""]:
                continue
            print(f"Updating {package}: {current} -> {latest}")
        lines[i] = f"{package}=={latest}\n"
        changed += 1

    with open(REQUIREMENTS_PATH, "w") as f:
        f.writelines(lines)

    print(f"Updated requirements.txt! {changed} package(s) changed.")


if __name__ == "__main__":
    try:
        args = parse_args()
        update_requirements(args.yes)
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)

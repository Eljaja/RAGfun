from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TEST_FILES = [
    "test_gateway_methods.py",
    "test_default_project_policy.py",
    "test_streaming_endpoints.py",
]


def main() -> int:
    tests_dir = Path(__file__).resolve().parent
    failures = []

    for name in TEST_FILES:
        path = tests_dir / name
        print(f"\n=== RUN {name} ===")
        proc = subprocess.run([sys.executable, str(path)], check=False)
        if proc.returncode != 0:
            failures.append(name)

    print("\n=== SUMMARY ===")
    if failures:
        print(f"Failed tests: {len(failures)}")
        for name in failures:
            print(f"- {name}")
        return 1

    print("All tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

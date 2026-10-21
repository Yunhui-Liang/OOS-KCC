"""Check release files against MANIFEST_SHA256.txt without external packages."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    lines = (ROOT / "MANIFEST_SHA256.txt").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(f"Checksum mismatch: {relative}")
    print(f"Verified {len(lines)} release files")


if __name__ == "__main__":
    main()

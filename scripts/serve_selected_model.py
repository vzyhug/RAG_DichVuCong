"""Start vLLM using the manifest produced by prepare_model_serving.py."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("deployment/selected_model.json"))
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    command = list(manifest["command"])
    api_key = os.getenv("VLLM_API_KEY", "").strip()
    if api_key:
        command.extend(["--api-key", api_key])
    print("Starting:", " ".join(command[:-1] + ["<redacted>"]) if api_key else " ".join(command))
    if args.print_only:
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())

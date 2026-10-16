#!/usr/bin/env bash
# Run every bash command in README.md. A line marked "# needs LLM_API_KEY" is skipped.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

commands=$(mktemp)
trap 'rm -f "$commands"' EXIT
uv run python - << 'PY' > "$commands"
import re
from pathlib import Path

text = Path("README.md").read_text()
blocks = re.findall(r"```bash\n(.*?)```", text, flags=re.S)
for block in blocks:
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "needs LLM_API_KEY" in stripped:
            continue
        print(stripped)
PY

while IFS= read -r cmd; do
  echo "+ $cmd"
  bash -lc "$cmd"
done < "$commands"

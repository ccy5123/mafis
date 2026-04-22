#!/usr/bin/env bash
# Create a 16K-context variant of llama3.1:8b without sudo.
# Phase 1B's Analyst prompt + value chain doc + tool schemas + reasoning trace
# exceed the 4096-token default. This variant raises num_ctx to 16384.
#
# Run once:
#   bash scripts/create_llama_16k.sh
#
# After it succeeds, update .env:
#   ANALYST_MODEL=llama3.1:8b-16k
#   VALUER_MODEL=llama3.1:8b-16k
set -euo pipefail

TMPDIR=$(mktemp -d)
cat > "$TMPDIR/Modelfile" <<'EOF'
FROM llama3.1:8b
PARAMETER num_ctx 16384
EOF

echo "Creating llama3.1:8b-16k …"
ollama create llama3.1:8b-16k -f "$TMPDIR/Modelfile"
rm -rf "$TMPDIR"
echo "Done. Installed models:"
ollama list

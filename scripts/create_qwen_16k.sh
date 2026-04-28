#!/usr/bin/env bash
# Create a 16K-context variant of qwen2.5:7b without sudo.
#
# Why this exists: the .env file routes ANALYST/VALUER/STEWARD/Stage 3 to
# `qwen2.5:7b-16k`, which is a custom Modelfile alias built on qwen2.5:7b
# with num_ctx raised from the 4096 default. The full Stage 3 prompt
# (axis definitions extracted verbatim from constitution.md + per-axis
# proxy details + JSON schema) routinely exceeds 4K tokens, and Stage 4's
# Defender prompt is even larger.
#
# Run once:
#   bash scripts/create_qwen_16k.sh
#
# Prerequisite: `ollama pull qwen2.5:7b` (the base model).
set -euo pipefail

if ! ollama list 2>/dev/null | grep -q '^qwen2.5:7b\s'; then
    echo "Base model qwen2.5:7b not found. Pulling..."
    ollama pull qwen2.5:7b
fi

TMPDIR=$(mktemp -d)
cat > "$TMPDIR/Modelfile" <<'EOF'
FROM qwen2.5:7b
PARAMETER num_ctx 16384
EOF

echo "Creating qwen2.5:7b-16k …"
ollama create qwen2.5:7b-16k -f "$TMPDIR/Modelfile"
rm -rf "$TMPDIR"
echo "Done. Installed models:"
ollama list

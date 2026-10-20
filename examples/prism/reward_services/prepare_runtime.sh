#!/usr/bin/env bash
# Run once in a networked CPU environment. The base interpreter must also be
# accessible from the Serving image (the portable runtime in INSPIRE.md works).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${RUNTIME_DIR:?Set RUNTIME_DIR to a new shared-storage virtualenv directory}"
BASE_PYTHON="${BASE_PYTHON:-/inspire/ssd/project/cq-scientific-cooperation-zone/public/kyu/venvs/py312-runtime/bin/python3.12}"
if [ -e "$RUNTIME_DIR" ]; then
  echo "Refusing to modify an existing runtime: $RUNTIME_DIR" >&2
  exit 1
fi
"$BASE_PYTHON" -m venv --copies "$RUNTIME_DIR"
"$RUNTIME_DIR/bin/python" -m pip install -r "$HERE/requirements-embeddings.txt"
if [ "${WITH_FLASH_ATTN:-1}" = 1 ]; then
  "$RUNTIME_DIR/bin/python" - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 12), "Pinned FA2 wheel requires Python 3.12"
assert torch.__version__.startswith("2.8."), "Pinned FA2 wheel requires Torch 2.8"
assert torch._C._GLIBCXX_USE_CXX11_ABI, "Pinned FA2 wheel requires the C++11 ABI"
assert torch.version.cuda and torch.version.cuda.startswith("12."), "CUDA 12 runtime required"
PY
  "$RUNTIME_DIR/bin/python" -m pip install --no-deps \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb'
  "$RUNTIME_DIR/bin/python" -c 'import flash_attn; from flash_attn.bert_padding import unpad_input; print("FA2", flash_attn.__version__)'
fi
"$RUNTIME_DIR/bin/python" -m pip freeze > "$RUNTIME_DIR/packages.txt"
echo "Set PYTHON=$RUNTIME_DIR/bin/python when launching either embedding service."

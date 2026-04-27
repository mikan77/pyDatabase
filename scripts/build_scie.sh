#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python3}
DIST_DIR=${DIST_DIR:-"$ROOT/dist"}
WHEELHOUSE_DIR=${WHEELHOUSE_DIR:-"$ROOT/wheelhouse"}
OUTPUT_PATH=${OUTPUT_PATH:-"$DIST_DIR/uspexdb-scie"}

echo "==> Project root: $ROOT"
echo "==> Python: $PYTHON_BIN"
echo "==> Output: $OUTPUT_PATH"

mkdir -p "$DIST_DIR" "$WHEELHOUSE_DIR"

MISSING_TOOLS=$(
  "$PYTHON_BIN" - <<'PY'
import importlib.util

required = ("build", "pex", "certifi", "wheel")
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(" ".join(missing))
PY
)

if [ -n "$MISSING_TOOLS" ]; then
  echo "==> Installing build tools: $MISSING_TOOLS"
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    "$PYTHON_BIN" -m pip install -U build wheel pex certifi
  else
    "$PYTHON_BIN" -m pip install --user -U build wheel pex certifi
  fi
fi

echo "==> Building project wheel"
"$PYTHON_BIN" -m build --wheel --no-isolation

WHEEL_PATH=$(ls -t "$DIST_DIR"/uspexdb-*.whl 2>/dev/null | head -n 1 || true)
if [ -z "$WHEEL_PATH" ] || [ ! -f "$WHEEL_PATH" ]; then
  echo "Wheel was not created in $DIST_DIR" >&2
  exit 1
fi

echo "==> Refreshing local wheelhouse from $WHEEL_PATH"
"$PYTHON_BIN" -m pip download --dest "$WHEELHOUSE_DIR" "$WHEEL_PATH"

CERT_FILE=$(
  "$PYTHON_BIN" - <<'PY'
import certifi
print(certifi.where())
PY
)

echo "==> Building PEX scie"
SSL_CERT_FILE="$CERT_FILE" \
  "$PYTHON_BIN" -m pex \
  "$WHEEL_PATH" \
  -f "$WHEELHOUSE_DIR" \
  --no-pypi \
  -c uspexdb \
  --scie eager \
  --scie-only \
  -o "$OUTPUT_PATH"

echo "==> Built: $OUTPUT_PATH"
file "$OUTPUT_PATH" || true

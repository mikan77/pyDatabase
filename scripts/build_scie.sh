#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python3}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-"$ROOT/artifacts"}
DIST_DIR=${DIST_DIR:-"$ROOT/dist"}
WHEELHOUSE_DIR=${WHEELHOUSE_DIR:-"$ARTIFACTS_DIR/wheelhouse"}
OUTPUT_PATH=${OUTPUT_PATH:-"$DIST_DIR/uspexdb-scie"}
PEX_PIP_VERSION=${PEX_PIP_VERSION:-23.2}
PEX_SETUPTOOLS_VERSION=${PEX_SETUPTOOLS_VERSION:-68.0.0}
PEX_WHEEL_VERSION=${PEX_WHEEL_VERSION:-0.40.0}

echo "==> Project root: $ROOT"
echo "==> Python: $PYTHON_BIN"
echo "==> Output: $OUTPUT_PATH"
echo "==> Artifacts: $ARTIFACTS_DIR"

mkdir -p "$DIST_DIR" "$ARTIFACTS_DIR" "$WHEELHOUSE_DIR"

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
"$PYTHON_BIN" -m build --wheel --no-isolation --outdir "$DIST_DIR"

WHEEL_PATH=$(ls -t "$DIST_DIR"/uspexdb-*.whl 2>/dev/null | head -n 1 || true)
if [ -z "$WHEEL_PATH" ] || [ ! -f "$WHEEL_PATH" ]; then
  echo "Wheel was not created in $DIST_DIR" >&2
  exit 1
fi

"$ROOT/scripts/prepare_wheelhouse.sh" "$WHEEL_PATH"

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
  --pip-version "$PEX_PIP_VERSION" \
  -c uspexdb \
  --scie eager \
  --scie-only \
  -o "$OUTPUT_PATH"

echo "==> Built: $OUTPUT_PATH"
file "$OUTPUT_PATH" || true

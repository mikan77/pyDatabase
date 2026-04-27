#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python3}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-"$ROOT/artifacts"}
WHEELHOUSE_DIR=${WHEELHOUSE_DIR:-"$ARTIFACTS_DIR/wheelhouse"}
DIST_DIR=${DIST_DIR:-"$ROOT/dist"}
PEX_PIP_VERSION=${PEX_PIP_VERSION:-23.2}
PEX_SETUPTOOLS_VERSION=${PEX_SETUPTOOLS_VERSION:-68.0.0}
PEX_WHEEL_VERSION=${PEX_WHEEL_VERSION:-0.40.0}

WHEEL_PATH=${1:-}
if [ -z "$WHEEL_PATH" ]; then
  WHEEL_PATH=$(ls -t "$DIST_DIR"/uspexdb-*.whl 2>/dev/null | head -n 1 || true)
fi

if [ -z "$WHEEL_PATH" ] || [ ! -f "$WHEEL_PATH" ]; then
  echo "Project wheel was not found. Pass the wheel path explicitly or build it first." >&2
  exit 1
fi

mkdir -p "$WHEELHOUSE_DIR"

echo "==> Wheelhouse directory: $WHEELHOUSE_DIR"
echo "==> Project wheel: $WHEEL_PATH"
echo "==> PEX bootstrap versions: pip=$PEX_PIP_VERSION setuptools=$PEX_SETUPTOOLS_VERSION wheel=$PEX_WHEEL_VERSION"

"$PYTHON_BIN" -m pip download \
  --dest "$WHEELHOUSE_DIR" \
  "$WHEEL_PATH" \
  "pip==$PEX_PIP_VERSION" \
  "setuptools==$PEX_SETUPTOOLS_VERSION" \
  "wheel==$PEX_WHEEL_VERSION"

find "$WHEELHOUSE_DIR" -maxdepth 1 \( -name '*.tar.gz' -o -name '*.zip' \) -print | while IFS= read -r sdist; do
  [ -n "$sdist" ] || continue
  echo "==> Building wheel from source archive: $sdist"
  "$PYTHON_BIN" -m pip wheel \
    --no-deps \
    --no-build-isolation \
    --wheel-dir "$WHEELHOUSE_DIR" \
    "$sdist"
done

echo "==> Wheelhouse is ready"

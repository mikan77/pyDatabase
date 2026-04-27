#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"

python3 -m pip install --editable .
python3 setup.py build_ext --inplace

#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export MVCAM_COMMON_RUNENV="${MVCAM_COMMON_RUNENV:-/opt/MVS/lib}"
export MVCAM_GENICAM_CLPROTOCOL="${MVCAM_GENICAM_CLPROTOCOL:-/opt/MVS/lib/CLProtocol}"
export LD_LIBRARY_PATH="${MVCAM_COMMON_RUNENV}/64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
exec .venv/bin/python "$@"

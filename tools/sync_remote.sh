#!/usr/bin/env bash
# Copy this checkout to the H100 box and install it editable there.
#   tools/sync_remote.sh            # sync + pip install -e
#   SKIP_INSTALL=1 tools/sync_remote.sh
# CUTLASS is not copied (third_party/cutlass): the first sync clones the pinned
# tag on the remote.
set -euo pipefail

HOST=${GREEN_REMOTE_HOST:-root@103.73.232.171}
PORT=${GREEN_REMOTE_PORT:-64061}
KEY=${GREEN_REMOTE_KEY:-$HOME/.ssh/id_rsa}
DST=${GREEN_REMOTE_DIR:-/network-volume/rnd-sglang/green/GreenAttention}
REPO=$(cd "$(dirname "$0")/.." && pwd)
CUTLASS_TAG=$(cat "$REPO/third_party/CUTLASS_VERSION")

ssh_remote() { ssh -o BatchMode=yes -p "$PORT" -i "$KEY" "$HOST" "$@"; }

COPYFILE_DISABLE=1 tar -C "$REPO" --no-xattrs -cz --exclude .git --exclude '._*' --exclude third_party/cutlass \
    --exclude '__pycache__' --exclude '*.so' . |
    ssh_remote "mkdir -p '$DST' && tar -xz -C '$DST'"

ssh_remote "set -e
cd '$DST'
if [ ! -f third_party/cutlass/include/cute/tensor.hpp ]; then
  rm -rf third_party/cutlass
  git -c advice.detachedHead=false clone -q --depth 1 --branch '$CUTLASS_TAG' https://github.com/NVIDIA/cutlass third_party/cutlass
fi
if [ -z '${SKIP_INSTALL:-}' ]; then
  /usr/bin/python -m pip install -q -e . --no-deps --no-build-isolation
fi"
echo "synced $REPO -> $HOST:$DST (cutlass $CUTLASS_TAG)"

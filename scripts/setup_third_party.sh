#!/usr/bin/env bash
# Clone the upstream checkouts EgoSurg calls. Nothing here is vendored.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-${ROOT}/third_party}"
mkdir -p "${DEST}"

clone_pin() {
  local url="$1" name="$2" commit="$3"
  if [[ ! -d "${DEST}/${name}/.git" ]]; then
    git clone "${url}" "${DEST}/${name}"
  fi
  git -C "${DEST}/${name}" fetch --depth 1 origin "${commit}"
  git -C "${DEST}/${name}" checkout --detach "${commit}"
}

clone_pin https://github.com/nv-tlabs/Difix3D.git Difix3D c76edc595586e16732c91ddee82f3a6d83a8a9cc
clone_pin https://github.com/NVlabs/FoundationStereo.git FoundationStereo 41390d52e157a75ee1be6103b5bd6f01dd4d14d5

echo "DIFIX3D_ROOT=${DEST}/Difix3D"
echo "FOUNDATION_STEREO_ROOT=${DEST}/FoundationStereo"
echo "Install gsplat from pip (v1.5.3). Do not use a vendored copy."

#!/usr/bin/env bash
# Export the images to a single tarball for moving onto a machine that may not
# want to build (or may not have the registry reachable).
#
#   bash docker/export-image.sh            harness only, ~250 MB
#   WITH_OLLAMA=1 bash docker/export-image.sh   harness + the Ollama runtime, ~2 GB
#
# Model weights are NOT included: they live in the named volume, are several GB,
# and are better pulled once on the target box than shipped in a tarball.
set -euo pipefail

OUT=${OUT:-dist}
HARNESS=${HARNESS_IMAGE:-wil-rag-harness:latest}
OLLAMA=${OLLAMA_IMAGE:-ollama/ollama:latest}

mkdir -p "$OUT"

docker image inspect "$HARNESS" >/dev/null 2>&1 || {
  echo "building $HARNESS first"; docker compose build harness; }

images=("$HARNESS")
if [[ "${WITH_OLLAMA:-0}" == "1" ]]; then
  docker image inspect "$OLLAMA" >/dev/null 2>&1 || docker pull "$OLLAMA"
  images+=("$OLLAMA")
fi

tar="$OUT/wil-rag-harness.tar"
echo "saving: ${images[*]}"
docker save "${images[@]}" -o "$tar"
gzip -f "$tar"

sha256sum "$tar.gz" 2>/dev/null || shasum -a 256 "$tar.gz"
ls -lh "$tar.gz"

cat <<TXT

Move $tar.gz to the Linux box, then:

  gunzip -c wil-rag-harness.tar.gz | docker load
  docker compose up -d ollama
  docker compose run --rm harness models     # one-time, pulls ~5 GB of weights
  docker compose run --rm --no-deps harness tests

You still need compose.yaml on the target machine — copy it across with the tarball.
TXT

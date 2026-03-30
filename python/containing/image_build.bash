set -euo pipefail

port=$(< port.txt)
echo "building image (port: $port)"

docker image build \
  --progress plain \
  --no-cache \
  --file ./Dockerfile \
  --build-arg PORT="$port" \
  --tag notebook_image \
  ../

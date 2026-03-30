set -euo pipefail
echo "starting container"
docker container start \
  --interactive \
  --attach \
  notebook_container

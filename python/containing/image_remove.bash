set -euo pipefail
echo "removing image"
docker image rm notebook_image --force

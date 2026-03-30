set -euo pipefail

port=$(< port.txt)
echo "creating container (port: $port)"

docker container create \
  --publish "$port:$port" \
  --name notebook_container \
  --interactive \
  notebook_image

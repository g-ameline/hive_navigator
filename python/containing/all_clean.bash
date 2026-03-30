set -euo pipefail
bash "./container_stop.sh"  || true
bash "./container_remove.sh" || true
bash "./image_remove.sh"     || true
docker system prune

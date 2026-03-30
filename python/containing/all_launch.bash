set -euo pipefail
bash "./image_build.sh"
bash "./container_create.sh"
bash "./container_start.sh"

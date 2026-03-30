set -euo pipefail
echo "extracting notebooks from container"
mkdir --parents ./extracted_notebooks
docker container cp \
  notebook_container:./notebooks \
  ./extracted_notebooks

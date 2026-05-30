#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-configs/mono/tum}"

CONFIG_FILES=(
  "fr1_desk.yaml"
  "fr2_xyz.yaml"
  "fr3_office.yaml"
)

for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
  echo "Running python slam.py --config $CONFIG_DIR/$CONFIG_FILE"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python slam.py --config "$CONFIG_DIR/$CONFIG_FILE"
done

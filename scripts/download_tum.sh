#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="datasets/tum"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

download_and_extract() {
  local url="$1"
  local archive="${url##*/}"
  local dirname="${archive%.tgz}"

  if [ -d "$dirname" ]; then
    echo "$dirname already exists, skipping."
    return
  fi

  if [ ! -f "$archive" ]; then
    wget "$url"
  fi

  tar -xvzf "$archive"
}

download_and_extract "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
download_and_extract "https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz"
download_and_extract "https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"

#!/bin/bash
set -e

IMAGE_NAME="active-site-gnn"
CONTAINER_NAME="active-site-gnn-run"

# build the image
docker build -t "$IMAGE_NAME" .

# remove any leftover container with the same name
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# run training, the trained model persists outside of the container
docker run --name "$CONTAINER_NAME" \
    -v "$(pwd)/data:/app/data" \
    "$IMAGE_NAME" "$@"

# clean the container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
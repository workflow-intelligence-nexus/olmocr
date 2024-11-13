#!/bin/bash

set -e

VERSION=$(python -c 'import pdelfin.version; print(pdelfin.version.VERSION)')
echo "$VERSION"

docker build --platform linux/amd64 -f ./scripts/beaker/Dockerfile-inference  -t pdelfin-inference-$VERSION .
beaker image create --workspace ai2/oe-data-pdf --name pdelfin-inference-$VERSION pdelfin-inference-$VERSION
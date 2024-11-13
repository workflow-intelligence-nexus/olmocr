#!/bin/bash

set -e

docker build --platform linux/amd64 -f ./scripts/beaker/Dockerfile-inference  -t pdelfin-inference .

beaker image create --workspace ai2/oe-data-pdf --name pdelfin-inference pdelfin-inference
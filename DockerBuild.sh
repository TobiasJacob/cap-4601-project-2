#!/bin/sh
docker build -t cap4601proj2dev .
docker build -t cap4601proj2devgpu2 -f Dockerfile.gpu2 .

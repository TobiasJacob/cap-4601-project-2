#!/bin/sh
docker run -it --rm -v $(pwd):/workdir cap4601proj2dev /bin/bash
docker run -it --rm -v $(pwd):/workdir cap4601proj2devgpu /bin/bash

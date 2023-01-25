#!/bin/sh

perf record -F 99 -a -g --  mpirun -np $1 ./bin/node_cli
perf script | ../FlameGraph/stackcollapse-perf.pl > /tmp/out.perf-folded
../FlameGraph/flamegraph.pl /tmp/out.perf-folded > flamegraph.svg
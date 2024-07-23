#!/bin/bash

DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cargo build --profile release-lto

STORE_DIR=$DIR/../../presets/stores
BENCHMARK=$DIR/../../presets/benchmarks/readpopular.toml

STORES="chashmap contrie dashmap flurry papaya scchashmap mutex_hashmap rwlock_hashmap"

for s in $STORES; do
    for t in `seq 1 16`; do
        env global.threads=$t cargo run --profile release-lto -- bench -s $STORE_DIR/$s.toml -b $BENCHMARK 2>/dev/null | tee -a $s.txt
    done
done

gnuplot $DIR/plot.gpl

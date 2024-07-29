#!/bin/bash

DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cargo +stable build --profile release-lto --all-features

STORE_DIR=$DIR/../../presets/stores
BENCHMARK=$DIR/mixed.toml

STORES="chashmap contrie dashmap flurry papaya scchashmap mutex_hashmap rwlock_hashmap"

for s in $STORES; do
    echo $s
    rm $s.txt 2>/dev/null
    env global.threads=32 \
        cargo +stable run --profile release-lto --all-features -- \
        bench -s $STORE_DIR/$s.toml -b $BENCHMARK 2>/dev/null | tee $s.txt
done

gnuplot $DIR/plot.gpl

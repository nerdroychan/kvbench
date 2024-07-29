#!/bin/bash

DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cargo +stable build --profile release-lto --all-features

STORE_DIR=$DIR/../../presets/stores
BENCHMARK=$DIR/readpopular.toml

STORES="chashmap contrie dashmap flurry papaya scchashmap mutex_hashmap rwlock_hashmap"

for s in $STORES; do
    echo $s
    rm $s.txt 2>/dev/null
    for t in `seq 1 16`; do
        data="$(env global.threads=$t cargo +stable run --profile release-lto --all-features -- bench -s $STORE_DIR/$s.toml -b $BENCHMARK 2>/dev/null)"
        echo "threads $t $data" | tee -a $s.txt
    done
done

gnuplot $DIR/plot.gpl

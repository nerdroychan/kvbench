#!/bin/bash

DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cargo +stable build --profile release-lto --all-features

STORE_DIR=$DIR/../../presets/stores

STORES="chashmap contrie dashmap flurry papaya scchashmap mutex_hashmap rwlock_hashmap"

for s in $STORES; do
    for b in readpopular writeheavy; do
        benchmark=$DIR/../$b/$b.toml
        echo $s-$b
        rm $s-$b.txt 2>/dev/null
        for t in `seq 1 16`; do
            data="$(env global.latency=true global.threads=$t cargo +stable run --profile release-lto --all-features -- bench -s $STORE_DIR/$s.toml -b $benchmark 2>/dev/null)"
            echo "threads $t $data" | tee -a $s-$b.txt
        done
    done
done

gnuplot $DIR/plot.gpl

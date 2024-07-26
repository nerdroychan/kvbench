#!/bin/bash

DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cargo build --profile release-lto --all-features

STORE_DIR=$DIR/../../presets/stores
BENCHMARK=$DIR/mixed.toml

STORES="chashmap contrie dashmap flurry papaya scchashmap mutex_hashmap rwlock_hashmap"

for s in $STORES; do
    env global.threads=32 cargo run --profile release-lto --all-features -- bench -s $STORE_DIR/$s.toml -b $BENCHMARK 2>/dev/null | tee $s.txt
done

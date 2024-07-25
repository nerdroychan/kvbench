This example shows how to integrate `kvbench` into your own key-value store implementations.

To compile, simply use:

```
cargo build --release
```

Then you can run the benchmark binary with this newly added key-value store:

```
./target/release/your-kv-store bench -s your-kv-store.toml -b benchmark.toml
```

Or you can also start a server on it:

```
./target/release/your-kv-store server -s your-kv-store.toml
```

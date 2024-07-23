# kvbench

A benchmarking framework designed for testing key-value stores with easily customizable
workloads.

With `kvbench`, you can define the details of a benchmark using the TOML format, such as the
proportions of mixed operations, the key access pattern, and key space size, just to name a
few. In addition to regular single-process benchmarks, `kvbench` also integrates a key-value
client/server implementation that works with a dedicated server thread/machine.

You can also incorporate `kvbench` into your own key-value store implementations and run it
against the built-in stores. All you need is implementing the [`KVMap`] or the [`AsyncKVMap`]
trait, depending on the type of the store. After registering your store, simply reuse the
exported [`cmdline()`] in your `main` function and it will work seamlessly with your own store.

More detailed usage could be found in the module-level rustdocs.

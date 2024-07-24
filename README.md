# kvbench

![GitHub Workflow](https://github.com/nerdroychan/kvbench/actions/workflows/test.yml/badge.svg)
![GPLv3](https://img.shields.io/github/license/nerdroychan/kvbench)

A benchmarking framework designed for testing key-value stores with easily customizable
workloads.

## Intro

This Rust crate enables the execution of tailored benchmarks on various key-value stores. Users
have the flexibility to adjust benchmark and key-value store parameters and store them in a
TOML-formatted file. The default command line interface is capable of loading these files and
running the benchmarks as specified.

In addition to standard single-process benchmarks, kvbench seamlessly incorporates a key-value
client/server setup that operates with a dedicated server thread/machine.

## Usage

Without any changes, you can run the default command line interface with three modes:

- `kvbench bench -s <STORE_FILE> -b <BENCH_FILE>` runs a (group of) benchmark using the parameters
stored in `<STORE_FILE>` and `<BENCH_FILE>`.
- `kvbench server -h <HOST> -p <PORT> -s <STORE_FILE> -n <THREADS>` boots up a key-value server
with `<THREADS>` workers, listening on `<HOST>:<PORT>`, and uses the key-value stores specified in
`<STORE_FILE>`.
- `kvbench list` lists all registered key-value stores that can be used.

See [examples](examples/) for more examples.

## Configuration

See the documentation of the modules `stores` and `bench` for available options.

## Integration

You can incorporate `kvbench` into your own key-value store implementations and run it
against the built-in stores. All you need is implementing the necessary traits, depending on the
type of the store, and call the default command line interface provided by this crate.

See [examples/your-kv-store](examples/your-kv-store) for a minimal but concrete example.

## Development

The missing pieces that are currently under active development:

- Latency measurement (incl. CDF and tail/avg. latency metrics).
- Atomic Read-modify-write (RMW) support.
- More key-distributions (e.g., latest key, composite-zipfian).
- Ordered key-value stores support (range query workloads).
- Extra built-ins (e.g., YCSB workloads).

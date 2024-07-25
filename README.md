# kvbench

![GitHub Workflow](https://github.com/nerdroychan/kvbench/actions/workflows/test.yml/badge.svg)
![GPLv3](https://img.shields.io/github/license/nerdroychan/kvbench)

A benchmark framework designed for testing key-value stores with easily customizable
workloads.

## Introduction

This Rust crate enables the execution of customizable benchmarks on various key-value stores.
Users have the flexibility to adjust benchmark and key-value store parameters and store them
in TOML-formatted files. The built-in command line interface is capable of loading these files and
running the benchmarks as specified.

In addition to standard single-process benchmarks, it also seamlessly incorporates a key-value
client/server implementation that operates with a dedicated server thread or machine.

## Usage

The [Documentation](https://docs.rs/kvbench) provides detailed usage guidelines.

## Examples

- [examples/your-kv-store](examples/your-kv-store): How to integrate `kvbench` into your own
key-value store implementations.
- [examples/readpopular](examples/readpopular): A benchmark that reads popular records in
the store, running with different number of threads.
- [examples/writeheavy](examples/writeheavy): A benchmark that mixes reads and writes at 1:1
on a random record in the store, running with different number of threads.
- [examples/mixed](examples/mixed): A benchmark that consists of multiple phases (5 seconds
running time each), benchmarked with 32 threads.

The built-in configuration files used by the above benchmarks can be found in [presets](presets).

## Development

The missing pieces that are currently under active development:

- Latency measurement (incl. CDF and tail/avg. latency metrics).
- Atomic Read-modify-write (RMW) support.
- More key-distributions (e.g., latest key, composite-zipfian).
- Ordered key-value stores support (range query workloads).
- Extra built-ins (e.g., YCSB workloads).

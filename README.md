# kvbench

[![Crates.io Version](https://img.shields.io/crates/v/kvbench)](https://crates.io/crates/kvbench/)
[![Docs.rs Status](https://img.shields.io/docsrs/kvbench)](https://docs.rs/kvbench/)

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

The [documentation](https://docs.rs/kvbench) provides detailed usage guidelines.

## Development

The missing pieces that are currently under active development:

- Latency measurement (incl. CDF and tail/avg. latency metrics).
- Atomic Read-modify-write (RMW) support.
- More key-distributions (e.g., latest key, composite-zipfian).
- Ordered key-value stores support (range query workloads).
- Extra built-ins (e.g., YCSB workloads).

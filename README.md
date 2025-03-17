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

This project is being actively developed. More built-in stores and benchmark parameters
are expected to be added.

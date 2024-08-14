#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! A benchmark framework designed for testing key-value stores with easily customizable
//! workloads.
//!
//! Key features:
//!
//! 1. Flexible and ergonomic control over benchmark specifications using TOML configuration files.
//! 2. Collecting diverse metrics, including throughput, latency (w/ CDF), and rate-limited latency.
//! 3. One-shot execution of multiple benchmark steps with different properties.
//! 4. Various built-in key-value stores in place as well as a client/server implementation.
//! 5. Highly extensible and can be seamlessly integrated into your own store.
//!
//! # Benchmark Configuration
//!
//! A benchmark in kvbench consists of one or more benchmark runs, termed as *phases*.
//! Phases will be run sequentially following their order in the configuration file.
//!
//! A benchmark configuration file is formatted in TOML. It consists of the definition of each
//! phase in an array named `benchmark`, so the configuration of each phase starts with
//! `[[benchmark]]`. The file also optionally contains a `[global]` section which will override the
//! unspecified field in each phase. This can eliminate redundant options in each phase, for
//! example, when those options are the same across the board.
//!
//! A configuration file generally looks like the following:
//!
//! ```toml
//! [global]
//! # global options
//!
//! [[benchmark]]
//! # phase 1 configuration
//!
//! [[benchmark]]
//! # phase 2 configuration
//!
//! ...
//! ```
//! Options in `[global]` section can also be overwritten via environment variables without
//! modifying the TOML file. For example, if the user needs to override `x` in `[global]`, one can
//! set the environment variable `global.x` (case insensitive). This is helpful when the user would
//! like to run different benchmarks when changing only a few options using a shell script.
//!
//! **Reference**
//!
//! - [`BenchmarkOpt`]: the available options for benchmark phase configuration.
//! - [`GlobalOpt`]: the available options for global configuration.
//!
//! # Key-Value Store Configuration
//!
//! In addition to the specification of the benchmark itself, kvbench also requires the
//! parameters of the key-value store it runs against. Only one key-value store runs at a time.
//!
//! The configuration of a key-value store is stored in a dictionary `map`.
//! A store's configuration file looks like the following:
//!
//! ```toml
//! [map]
//! name = "..."
//! # option1 = ...
//! # option2 = ...
//!
//! ...
//! ```
//! The field `name` must be given and it should be equal to the name registered by the store.
//! Other than `name`, all the fields are parsed as a string map and will be passed to the
//! store's constructor function. The options in `[map]` section can also be overwritten via
//! environment variables (e.g., setting `map.x` overrides property `x`).
//!
//! **Reference**
//!
//! - [`mod@stores`]: the available options for built-in stores and how to register new stores.
//!
//! # Run a Benchmark
//!
//! Once the configuration files of the benchmark along with the key-value store are ready, a
//! benchmark can be started by using the `bench` mode of the built-in command-line interface.
//!
//! **Reference**
//!
//! - [`cmdline()`]: the usage of the default command-line interface.
//!
//! # Metrics Collection
//!
//! Currently, all outputs are in plain text format. This makes the output easy to process using
//! shell scripts and tools including gnuplot. If there are new data added to the output, it
//! will be appended at the end of existing entries (but before `cdf` if it exists, see below)
//! to make sure outputs from old versions can still be processed without changes.
//!
//! ## Throughput-only Output (default case)
//!
//! When measuring throughput, an output may look like the following:
//! ```txt
//! phase 0 repeat 0 duration 1.00 elapsed 1.00 total 1000000 mops 1.00
//! phase 0 repeat 1 duration 1.00 elapsed 2.00 total 1000000 mops 1.00
//! phase 0 repeat 2 duration 1.00 elapsed 3.00 total 1000000 mops 1.00
//! phase 0 finish . duration 1.00 elapsed 3.00 total 3000000 mops 1.00
//! ```
//!
//! The general format is:
//!
//! ```txt
//! phase <p> repeat <r> duration <d> elapsed <e> total <o> mops <t>
//! ```
//!
//! Where:
//!
//! - `<p>`: phase id.
//! - `<r>`: repeat id in a phase, or string `finish .`, if the line is the aggregated report
//! of a whole phase.
//! - `<d>`: the duration of the repeat/phase, in seconds.
//! - `<e>`: the total elapsed seconds since the starting of the program.
//! - `<o>`: the total key-value operations executed by all worker threads in the repeat/phase.
//! - `<t>`: followed by the throughput in million operations per second of the repeat/phase.
//!
//! ## Throughput + Latency Output (when `latency` is `true`)
//!
//! When latency measurement is enabled, the latency metrics shall be printed at the end of each
//! benchmark. It is not shown after each repeat, because unlike throughput which is a singleton
//! value at a given time, latency is a set of values and it usually matters only when we aggregate
//! a lot of them. The output format in this case is generally the same as throughput-only
//! measurements, but the `finish` line has extra output like the following:
//!
//! ```txt
//! phase 0 repeat 0 duration 1.00 elapsed 1.00 total 1000000 mops 1.00
//! phase 0 repeat 1 duration 1.00 elapsed 2.00 total 1000000 mops 1.00
//! phase 0 repeat 2 duration 1.00 elapsed 3.00 total 1000000 mops 1.00
//! phase 0 finish . duration 1.00 elapsed 3.00 total 3000000 mops 1.00 min_us 0.05 max_us 100.00 avg_us 50.00 p50_us 50.00 p95_us 95.00 p99_us 99.00 p999_us 100.00
//! ```
//!
//! The extra output on the last line has a format of:
//!
//! ```txt
//! min_us <i> max_us <a> avg_us <v> p50_us <m> p95_us <n> p99_us <p> p999_us <t>
//! ```
//!
//! Where (all units are microseconds):
//!
//! - `<i>`: minimum latency
//! - `<a>`: maximum latency
//! - `<v>`: mean latency
//! - `<m>`: median latency (50% percentile)
//! - `<n>`: P95 latency
//! - `<p>`: P99 latency
//! - `<t>`: P999 latency (99.9%)
//!
//! ## Throughput + Latency + Latency CDF Mode (when both `latency` and `cdf` are `true`)
//!
//! When `cdf` is enabled, the latency CDF data will be printed at the end of the same line as the
//! latency metrics above. In that case, the output will be like the following:
//!
//! ```txt
//! phase 0 repeat 0 duration 1.00 elapsed 1.00 total 1000000 mops 1.00
//! phase 0 repeat 1 duration 1.00 elapsed 2.00 total 1000000 mops 1.00
//! phase 0 repeat 2 duration 1.00 elapsed 3.00 total 1000000 mops 1.00
//! phase 0 finish . duration 1.00 elapsed 3.00 total 3000000 mops 1.00 min_us 0.05 max_us 100.00 avg_us 50.00 p50_us 50.00 p95_us 95.00 p99_us 99.00 p999_us 100.00 cdf_us percentile ...
//! ```
//! Since the latency metrics vary a lot between different benchmarks/runs, the number of data
//! points of the CDF is different. Therefore, it is printed at the end of the output only. It is
//! printed as a tuple of `<us> <percentile>` where `<us>` is the latency in microseconds and
//! `<percentile>` is the percentile of the accumulated operations with latency higher than between
//! `<us> - 1` and `<us>`, inclusively, ranging from 0 to 100 (two digit precision).
//! There can be arbitrary number of tuples. The output ends when the maximum recorded latency is
//! reached.
//!
//! An example of the CDF data will look like:
//!
//! ```txt
//! cdf_us percentile 1 0.00 2 0.00 3 0.00 4 10.00 5 20.00 6 20.00 ...
//! ```
//!
//! It means there are not data points at 1/2/3 microseconds. At 4 microseconds, there are 10% data
//! points. At 5 microseconds, there are another 10% data points which makes the total percentile
//! 20.00. At 6 microseconds, there are no data points so the percentile is still 20.00. Users can
//! post-process the output and make a smooth CDF plot out of it.
//!
//! # Server Mode
//! A key-value client/server implementation is available in kvbench. The server can be backed by
//! an arbitrary key-value store defined by a TOML file as in a benchmark, and the server can be
//! started using the `server` mode of the built-in command-line interface.
//!
//! To benchmark the server's performance, users can use the built-in client implementation.
//!
//! **Reference**
//!
//! - [`cmdline()`]: the usage of the default command-line interface.
//! - [`stores::remote`]: the available options of the key-value store client.

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

/// A synchronous, thread-safe key-value store.
///
/// This trait is used for owned stores, with which a per-thread handle can be created. The default
/// benchmark/server implementation is provided, unless the use case needs to use specific thread
/// management implementations.
pub trait KVMap: Send + Sync + 'static {
    /// Create a handle that can be referenced by different threads in the system.
    /// For most stores, this can just be done using an Arc.
    fn handle(&self) -> Box<dyn KVMapHandle>;

    fn thread(&self) -> Box<dyn crate::thread::Thread> {
        Box::new(self::thread::DefaultThread)
    }
}

/// A per-thread handle that references a [`KVMap`].
///
/// The handle is the real object that exposes a key-value interface.
pub trait KVMapHandle {
    /// Adding a new key-value pair or blindly updating an existing key's value.
    fn set(&mut self, key: &[u8], value: &[u8]);

    /// Retrieving the value of a key if it exists.
    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>>;

    /// Removing a key if it exists.
    fn delete(&mut self, key: &[u8]);

    /// Querying a range starting from the first key greater than or equal to the given key.
    fn scan(&mut self, key: &[u8], n: usize) -> Vec<(Box<[u8]>, Box<[u8]>)>;
}

/// A single operation that is applied to the key-value store.
///
/// This trait is used mainly in [`AsyncKVMap`] and server/client implementation.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub enum Operation {
    /// Adding a new key-value pair or blindly updating an existing key's value.
    Set { key: Box<[u8]>, value: Box<[u8]> },

    /// Retrieving the value of a key if it exists.
    Get { key: Box<[u8]> },

    /// Removing a key if it exists.
    Delete { key: Box<[u8]> },

    /// Querying a range starting from the first key greater than or equal to the given key.
    Scan { key: Box<[u8]>, n: usize },
}

/// A request submitted by an asynchronous store.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Request {
    /// The (usually unique) identifier of the request, or custom data.
    pub id: usize,

    /// The real payload that contains the operation.
    pub op: Operation,
}

/// A response received by an asynchronous store.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Response {
    /// The `id` of the corresponding request.
    pub id: usize,

    /// The real payload that contains the potential returned value.
    ///
    /// - For a `SET` or `DELETE` request, this should be `None`.
    /// - For a `GET` request, this should contain the value of the key (single element).
    /// - For a `SCAN` request, this should contain the sequence of the range query results, ordered
    ///   like `key0, value0, key1, value1 ...`.
    pub data: Option<Vec<Box<[u8]>>>,
}

/// A non-blocking, thread-safe key-value map.
///
/// Unlike [`KVMap`], [`AsyncKVMap`] works in request/response style. Where each handle needs to be
/// created by registering an explicit responder that serves as the "callback" when the underlying
/// routine produces a response.
///
/// Specifically, for benchmark, each worker thread maintains just one handle, so the buffer is
/// per-worker. For server, each worker may manage multiple connections. Each connection needs its
/// own responder (we should not mix responses for different connections, obviously). Therefore, it
/// creates a handle for each incoming connection, and maintains a responder for it.
pub trait AsyncKVMap: Sync + Send + 'static {
    /// Create a handle that can be referenced by different threads in the system. Each handle
    /// corresponds to a shared `responder` that implements [`AsyncResponder`].
    fn handle(&self, responder: Rc<dyn AsyncResponder>) -> Box<dyn AsyncKVMapHandle>;

    fn thread(&self) -> Box<dyn crate::thread::Thread> {
        Box::new(self::thread::DefaultThread)
    }
}

/// A per-thread handle that references a [`AsyncKVMap`].
///
/// The handle is the real object that exposes a key-value interface.
pub trait AsyncKVMapHandle {
    /// Submit a batch of requests to the store and immediately return without responses.
    fn submit(&mut self, requests: &Vec<Request>);

    /// Try to fill more responses into the registered responder. This operation can be, for
    /// example, yield more CPU time for the worker thread to handle requests, or flush a buffer to
    /// get more responses.
    fn drain(&mut self);
}

/// An asynchronous entry point to callback when a response returns.
pub trait AsyncResponder {
    /// Whenever a new response is returned, this method is called with a [`Response`] moved.
    fn callback(&self, response: Response);
}

impl AsyncResponder for RefCell<Vec<Response>> {
    fn callback(&self, response: Response) {
        self.borrow_mut().push(response);
    }
}

mod bench;
mod cmdline;
mod server;
pub mod stores;
pub mod thread;
mod workload;

pub use bench::{BenchmarkOpt, GlobalOpt};
pub use cmdline::cmdline;
pub use workload::WorkloadOpt;

pub extern crate inventory;
pub extern crate toml;

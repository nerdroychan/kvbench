#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! A benchmark framework designed for testing key-value stores with easily customizable
//! workloads.
//!
//! With `kvbench`, you can define the details of a benchmark using the TOML format, such as the
//! proportions of mixed operations, the key access pattern, and key space size, just to name a
//! few. In addition to regular single-process benchmarks, `kvbench` also integrates a key-value
//! client/server implementation that works with a dedicated server thread/machine.
//!
//! You can also incorporate `kvbench` into your own key-value store implementations and run it
//! against the built-in stores. All you need is implementing the [`KVMap`] or the [`AsyncKVMap`]
//! trait, depending on the type of the store. After registering your store, simply reuse the
//! exported [`cmdline()`] in your `main` function and it will work seamlessly with your own store.
//!
//! A few key design choices include:
//!
//! - Each key-value store exclusively stores a single type of key/value pair: variable-sized byte
//! arrays represented as [`u8`] slices on the heap. No generics over the key's type.
//! - The key-value store and the benchmark configurations are black boxes. They are created
//! dynamically from a TOML file, and dynamically dispatched.
//! - Benchmark functionalities can be reused in users' own crates: new key-value stores can be
//! dynamically registered without touching the source code of this crate.
//!
//! More detailed usage could be found in the module-level rustdocs:
//!
//! - [`mod@bench`] for the config format of a benchmark.
//! - [`mod@stores`] for the config format of a built-in key-value store.
//! - [`cmdline()`] for the usage of the default command line interface.

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

/// A request sent by a client to a server.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Request {
    /// The (usually unique) identifier of the request, or custom data.
    pub id: usize,

    /// The real payload that contains the operation.
    pub op: Operation,
}

/// A response sent by a server to a client.
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

pub mod bench;
mod cmdline;
pub mod server;
pub mod stores;
pub mod thread;
pub mod workload;

pub use cmdline::cmdline;

pub extern crate inventory;
pub extern crate toml;

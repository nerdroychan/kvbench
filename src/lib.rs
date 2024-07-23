//! A benchmarking framework designed for testing key-value stores with easily customizable
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
//! More detailed usage could be found in the module-level rustdocs.

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, Sender};
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

    /// The main bench method, with its default implementation usually doesn't need manual
    /// implementation unless the implementor needs custom thread spawn-join functions.
    /// If one would like to manually implement this method, it is needed to explicitly declare a
    /// new [`thread::Thread`] object and pass it to [`bench::bench_regular`].
    fn bench(self: Box<Self>, phases: &Vec<Arc<crate::bench::Benchmark>>) {
        let map = Arc::new(self);
        let thread = crate::thread::DefaultThread;
        crate::bench::bench_regular(map, phases, thread);
    }

    /// Start the main loop of KV server while using this map as the backend. There is no need to
    /// manually implement this method unless the implementor needs custom thread spawn-join
    /// functions. If one would like to manually implement this method, it is needed to explicitly
    /// declare a new [`thread::Thread`] object and pass it to [`server::server_regular`].
    fn server(
        self: Box<Self>,
        host: &str,
        port: &str,
        nr_workers: usize,
        stop_rx: Receiver<()>,
        grace_tx: Sender<()>,
    ) {
        let map = Arc::new(self);
        let thread = crate::thread::DefaultThread;
        crate::server::server_regular(map, host, port, nr_workers, stop_rx, grace_tx, thread);
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

    // fn read_modify_write(&mut self, key: &[u8]);
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
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct Response {
    /// The `id` of the corresponding request.
    pub id: usize,

    /// The real payload that contains the potential returned value.
    pub data: Option<Box<[u8]>>,
}

/// An non-blocking, thread-safe key-value map.
///
/// Unlike [`KVMap`], [`AsyncKVMap`] works in request/response style. Where each handle needs to be
/// created by registering an explicit responder that serves as the "callback" when the underlying
/// routine produces a response.
///
/// Specifically, for benchmark, each worker thread maintains just one handle, so the buffer is
/// per-worker. For server, each worker may manage multiple coneections. Each connection needs its
/// own responder (we should not mix responses for different connections, obviously). Therefore, it
/// creates a handle for each incoming connection, and maintains a responder for it.
pub trait AsyncKVMap: Sync + Send + 'static {
    /// Create a handle that can be referenced by different threads in the system. Each handle
    /// corresponds to a shared `responder` that implements [`AsyncResponder`].
    fn handle(&self, responder: Rc<dyn AsyncResponder>) -> Box<dyn AsyncKVMapHandle>;

    /// Similar to [`KVMap::bench`], but calls [`bench::bench_async`] instead.
    fn bench(self: Box<Self>, phases: &Vec<Arc<crate::bench::Benchmark>>) {
        let map = Arc::new(self);
        let thread = crate::thread::DefaultThread;
        crate::bench::bench_async(map, phases, thread);
    }

    /// Similar to [`KVMap::server`], but calls [`server::server_async`] instead.
    fn server(
        self: Box<Self>,
        host: &str,
        port: &str,
        nr_workers: usize,
        stop_rx: Receiver<()>,
        grace_tx: Sender<()>,
    ) {
        let map = Arc::new(self);
        let thread = crate::thread::DefaultThread;
        crate::server::server_async(map, host, port, nr_workers, stop_rx, grace_tx, thread);
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
mod serialization;
pub mod server;
pub mod stores;
pub mod thread;
pub mod workload;

pub use cmdline::cmdline;

pub extern crate inventory;
pub extern crate toml;

#![feature(vec_into_raw_parts)]
#![feature(ptr_metadata)]

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

/// A synchronous, thread-safe key-value map. This trait is used for owned stores, with which a
/// per-thread handle can be created. The default benchmark/server implementation is provided,
/// unless the use case needs to use specific thread management implementations.
pub trait KVMap: Send + Sync + 'static {
    /// Create a handle that can be referenced by different threads in the system.
    /// For most kvmap, this can just be done using an Arc.
    fn handle(&self) -> Box<dyn KVMapHandle>;

    /// The main bench function, with its default implementation usually doesn't need manual
    /// implementation unless the implementor needs custom thread spawn-join functions.
    fn bench(self: Box<Self>, phases: &Vec<Arc<crate::bench::Benchmark>>) {
        let map = Arc::new(self);
        let thread = crate::thread::DefaultThread;
        crate::bench::bench_regular(map, phases, thread);
    }

    /// Start the main loop of KV server while using this map as the backend.
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

pub trait KVMapHandle {
    /// Adding a new key-value pair or blindly updating an existing key's value.
    fn set(&mut self, key: &[u8], value: &[u8]);

    /// Retrieving the value of a key.
    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>>;
}

/// Operation is one type of data accesses to a kvmap.
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub enum Operation {
    Set { key: Box<[u8]>, value: Box<[u8]> },
    Get { key: Box<[u8]> },
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Request {
    pub id: usize,
    pub op: Operation,
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct Response {
    pub id: usize,
    pub data: Option<Box<[u8]>>,
}

/// An asynchronous, thread-safe key-value map. Unlike `KVMap`, `AsyncKVMap` works with
/// request/response style. Where each handle needs to be created by registering an explicit
/// responder that serves as the "callback" when the underlying routine produces a response.
///
/// Specifically, for benchmark, each worker thread maintains just one handle, so the buffer is
/// per-worker. For server, each worker may manage multiple coneections. Each connection needs its
/// own responder (we should not mix responses for different connections, obviously). Therefore, it
/// creates a handle for each incoming connection, and maintains a responder for it.
pub trait AsyncKVMap: Sync + Send + 'static {
    fn handle(&self, responder: Rc<dyn AsyncResponder>) -> Box<dyn AsyncKVMapHandle>;

    fn bench(self: Box<Self>, phases: &Vec<Arc<crate::bench::Benchmark>>) {
        let map = Arc::new(self);
        let thread = crate::thread::DefaultThread;
        crate::bench::bench_async(map, phases, thread);
    }

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

pub trait AsyncKVMapHandle {
    fn submit(&mut self, requests: &Vec<Request>);

    fn drain(&mut self);
}

pub trait AsyncResponder {
    fn callback(&self, response: Response);
}

impl AsyncResponder for RefCell<Vec<Response>> {
    fn callback(&self, response: Response) {
        self.borrow_mut().push(response);
    }
}

pub mod bench;
pub mod client;
pub mod map;
pub mod serialization;
pub mod server;
pub mod thread;
pub mod workload;

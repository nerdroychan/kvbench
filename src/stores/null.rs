//! A store that does nothing. It can be used to measure overheads of the crate.
//!
//! ## Configuration Format
//!
//! ### Regular
//!
//! ``` toml
//! [map]
//! name = "nullmap"
//! ```
//!
//! This store is [`KVMap`].
//!
//! ### Async
//!
//! ``` toml
//! [map]
//! name = "nullmap_async"
//! ```
//!
//! This store is [`AsyncKVMap`].

use crate::stores::{BenchKVMap, Registry};
use crate::*;

#[derive(Clone)]
pub struct NullMap;

impl NullMap {
    pub fn new() -> Self {
        Self
    }

    pub fn new_benchkvmap(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Regular(Arc::new(Box::new(Self::new())))
    }

    pub fn new_benchkvmap_async(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Async(Arc::new(Box::new(Self::new())))
    }
}

impl KVMap for NullMap {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for NullMap {
    fn set(&mut self, _key: &[u8], _value: &[u8]) {}

    fn get(&mut self, _key: &[u8]) -> Option<Box<[u8]>> {
        None
    }

    fn delete(&mut self, _key: &[u8]) {}

    fn scan(&mut self, _key: &[u8], _n: usize) -> Vec<(Box<[u8]>, Box<[u8]>)> {
        Vec::new()
    }
}

inventory::submit! {
    Registry::new("nullmap", NullMap::new_benchkvmap)
}

struct NullMapAsyncHandle(Vec<usize>, Rc<dyn AsyncResponder>);

impl AsyncKVMap for NullMap {
    fn handle(&self, responder: Rc<dyn AsyncResponder>) -> Box<dyn AsyncKVMapHandle> {
        Box::new(NullMapAsyncHandle(Vec::new(), responder.clone()))
    }
}

impl AsyncKVMapHandle for NullMapAsyncHandle {
    fn drain(&mut self) {
        for i in self.0.iter() {
            self.1.callback(Response { id: *i, data: None });
        }
        self.0.clear();
    }

    fn submit(&mut self, requests: &Vec<Request>) {
        for r in requests.iter() {
            self.0.push(r.id);
        }
    }
}

inventory::submit! {
    Registry::new("nullmap_async", NullMap::new_benchkvmap_async)
}

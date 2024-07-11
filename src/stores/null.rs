use crate::bench::{BenchKVMap, Registry};
use crate::*;

/// NullMap does nothing. It can be used to measure overheads in the future.
#[derive(Clone)]
pub struct NullMap;

impl NullMap {
    pub fn new() -> Self {
        Self
    }

    pub fn new_benchkvmap(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Regular(Box::new(Self::new()))
    }

    pub fn new_benchkvmap_async(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Async(Box::new(Self::new()))
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
}

inventory::submit! {
    Registry::new("nullmap", NullMap::new_benchkvmap)
}

struct NullMapAsyncHandle(usize, Rc<dyn AsyncResponder>);

impl AsyncKVMap for NullMap {
    fn handle(&self, responder: Rc<dyn AsyncResponder>) -> Box<dyn AsyncKVMapHandle> {
        Box::new(NullMapAsyncHandle(0, responder.clone()))
    }
}

impl AsyncKVMapHandle for NullMapAsyncHandle {
    fn drain(&mut self) {
        let n = self.0;
        for _ in 0..n {
            self.1.callback(Response { id: 0, data: None });
        }
        self.0 -= n;
    }

    fn submit(&mut self, requests: &Vec<Request>) {
        let n = requests.len();
        self.0 += n;
    }
}

inventory::submit! {
    Registry::new("nullmap_async", NullMap::new_benchkvmap_async)
}

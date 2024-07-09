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

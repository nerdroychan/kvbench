//! Adapter implementation of [`scc::hash_map::HashMap`].
//!
//! ## Configuration Format
//!
//! ``` toml
//! [map]
//! name = "scchashmap"
//! ```
//! This store is [`KVMap`].

use crate::stores::{BenchKVMap, Registry};
use crate::*;

#[derive(Clone)]
pub struct SccHashMap(Arc<scc::hash_map::HashMap<Box<[u8]>, Box<[u8]>>>);

impl SccHashMap {
    pub fn new() -> Self {
        Self(Arc::new(
            scc::hash_map::HashMap::<Box<[u8]>, Box<[u8]>>::new(),
        ))
    }

    pub fn new_benchkvmap(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Regular(Box::new(Self::new()))
    }
}

impl KVMap for SccHashMap {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for SccHashMap {
    fn set(&mut self, key: &[u8], value: &[u8]) {
        if let Err(_) = self.0.insert(key.into(), value.into()) {
            assert!(self.0.update(key, |_, v| *v = value.into()).is_some());
        }
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        match self.0.get(key) {
            Some(r) => Some(r.clone()),
            None => None,
        }
    }

    fn delete(&mut self, key: &[u8]) {
        self.0.remove(key);
    }
}

inventory::submit! {
    Registry::new("scchashmap", SccHashMap::new_benchkvmap)
}

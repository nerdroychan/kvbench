//! Adapter implementation of [`chashmap::CHashMap`].
//!
//! ## Configuration Format
//!
//! ``` toml
//! [map]
//! name = "chashmap"
//! ```
//! This store is [`KVMap`].

use crate::stores::{BenchKVMap, Registry};
use crate::*;

#[derive(Clone)]
pub struct CHashMap(Arc<chashmap::CHashMap<Box<[u8]>, Box<[u8]>>>);

impl CHashMap {
    pub fn new() -> Self {
        Self(Arc::new(chashmap::CHashMap::<Box<[u8]>, Box<[u8]>>::new()))
    }

    pub fn new_benchkvmap(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Regular(Box::new(Self::new()))
    }
}

impl KVMap for CHashMap {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for CHashMap {
    fn set(&mut self, key: &[u8], value: &[u8]) {
        self.0.insert(key.into(), value.into());
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
    Registry::new("chashmap", CHashMap::new_benchkvmap)
}

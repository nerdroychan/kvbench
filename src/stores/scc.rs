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
        BenchKVMap::Regular(Arc::new(Box::new(Self::new())))
    }
}

impl KVMap for SccHashMap {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for SccHashMap {
    fn set(&mut self, key: &[u8], value: &[u8]) {
        match self.0.entry_sync(key.into()) {
            scc::hash_map::Entry::Occupied(mut o) => {
                *o.get_mut() = value.into();
            }
            scc::hash_map::Entry::Vacant(v) => {
                v.insert_entry(value.into());
            }
        }
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        self.0.read_sync(key, |_, r| r.clone())
    }

    fn delete(&mut self, key: &[u8]) {
        self.0.remove_sync(key);
    }

    fn scan(&mut self, _key: &[u8], _n: usize) -> Vec<(Box<[u8]>, Box<[u8]>)> {
        unimplemented!("Range query is not supported");
    }
}

inventory::submit! {
    Registry::new("scchashmap", SccHashMap::new_benchkvmap)
}

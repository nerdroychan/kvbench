//! How to add your implementation to `kvbench`.

extern crate kvbench;

use kvbench::inventory;
use kvbench::toml;

use kvbench::stores::{BenchKVMap, Registry};
use kvbench::{KVMap, KVMapHandle};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct YourKVMap(Arc<RwLock<HashMap<Box<[u8]>, Box<[u8]>>>>);

impl YourKVMap {
    pub fn new() -> Self {
        Self(Arc::new(
            RwLock::new(HashMap::<Box<[u8]>, Box<[u8]>>::new()),
        ))
    }

    pub fn new_benchkvmap(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Regular(Arc::new(Box::new(Self::new())))
    }
}

impl KVMap for YourKVMap {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for YourKVMap {
    fn set(&mut self, key: &[u8], value: &[u8]) {
        self.0.write().unwrap().insert(key.into(), value.into());
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        match self.0.read().unwrap().get(key) {
            Some(v) => Some(v.clone()),
            None => None,
        }
    }

    fn delete(&mut self, key: &[u8]) {
        self.0.write().unwrap().remove(key);
    }

    fn scan(&mut self, _key: &[u8], _n: usize) -> Vec<(Box<[u8]>, Box<[u8]>)> {
        unimplemented!();
    }
}

inventory::submit! {
    Registry::new("your_kv_store", YourKVMap::new_benchkvmap)
}

fn main() {
    // Call the `cmdline()` function directly here, and you will get the same benchmark binary
    // that contains your kv and all the built-in stores in `kvbench`.
    kvbench::cmdline();
}

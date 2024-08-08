//! Adapter implementation of [`flurry::HashMap`].
//!
//! ## Configuration Format
//!
//! ``` toml
//! [map]
//! name = "flurry"
//! ```
//!
//! This store is [`KVMap`].

use crate::stores::{BenchKVMap, Registry};
use crate::*;

#[derive(Clone)]
pub struct Flurry(Arc<flurry::HashMap<Box<[u8]>, Box<[u8]>>>);

impl Flurry {
    pub fn new() -> Self {
        Self(Arc::new(flurry::HashMap::<Box<[u8]>, Box<[u8]>>::new()))
    }

    pub fn new_benchkvmap(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Regular(Arc::new(Box::new(Self::new())))
    }
}

impl KVMap for Flurry {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for Flurry {
    fn set(&mut self, key: &[u8], value: &[u8]) {
        self.0.pin().insert(key.into(), value.into());
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        match self.0.pin().get(key) {
            Some(v) => Some(v.clone()),
            None => None,
        }
    }

    fn delete(&mut self, key: &[u8]) {
        self.0.pin().remove(key);
    }

    fn scan(&mut self, _key: &[u8], _n: usize) -> Vec<(Box<[u8]>, Box<[u8]>)> {
        unimplemented!("Range query is not supported");
    }
}

inventory::submit! {
    Registry::new("flurry", Flurry::new_benchkvmap)
}

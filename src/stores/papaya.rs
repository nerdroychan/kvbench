//! Adapter implementation of [`papaya::HashMap`].
//!
//! ## Configuration Format
//!
//! ``` toml
//! [map]
//! name = "papaya"
//! ```
//!
//! This store is [`KVMap`].

use crate::stores::{BenchKVMap, Registry};
use crate::*;

#[derive(Clone)]
pub struct Papaya(Arc<papaya::HashMap<Box<[u8]>, Box<[u8]>>>);

impl Papaya {
    pub fn new() -> Self {
        Self(Arc::new(papaya::HashMap::<Box<[u8]>, Box<[u8]>>::new()))
    }

    pub fn new_benchkvmap(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Regular(Box::new(Self::new()))
    }
}

impl KVMap for Papaya {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for Papaya {
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

    fn scan(&mut self, _key: &[u8], _n: usize) -> Vec<Box<[u8]>> {
        unimplemented!("Range query is not supported");
    }
}

inventory::submit! {
    Registry::new("papaya", Papaya::new_benchkvmap)
}

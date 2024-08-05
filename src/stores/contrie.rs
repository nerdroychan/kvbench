//! Adapter implementation of [`contrie::ConMap`].
//!
//! ## Configuration Format
//!
//! ``` toml
//! [map]
//! name = "contrie"
//! ```
//!
//! This store is [`KVMap`].

use crate::stores::{BenchKVMap, Registry};
use crate::*;

#[derive(Clone)]
pub struct Contrie(Arc<contrie::ConMap<Box<[u8]>, Box<[u8]>>>);

impl Contrie {
    pub fn new() -> Self {
        Self(Arc::new(contrie::ConMap::<Box<[u8]>, Box<[u8]>>::new()))
    }

    pub fn new_benchkvmap(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Regular(Box::new(Self::new()))
    }
}

impl KVMap for Contrie {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for Contrie {
    fn set(&mut self, key: &[u8], value: &[u8]) {
        self.0.insert(key.into(), value.into());
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        match self.0.get(key) {
            Some(r) => Some(r.value().clone()),
            None => None,
        }
    }

    fn delete(&mut self, key: &[u8]) {
        self.0.remove(key);
    }

    fn scan(&mut self, _key: &[u8], _n: usize) -> Vec<Box<[u8]>> {
        unimplemented!("Range query is not supported");
    }
}

inventory::submit! {
    Registry::new("contrie", Contrie::new_benchkvmap)
}

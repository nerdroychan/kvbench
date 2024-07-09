use crate::bench::{BenchKVMap, Registry};
use crate::*;

/// DashMap is a monolithic, lock-free concurrent hash map based on dashmap
#[derive(Clone)]
pub struct DashMap(Arc<dashmap::DashMap<Box<[u8]>, Box<[u8]>>>);

impl DashMap {
    pub fn new() -> Self {
        Self(Arc::new(dashmap::DashMap::<Box<[u8]>, Box<[u8]>>::new()))
    }

    pub fn new_benchkvmap(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Regular(Box::new(Self::new()))
    }
}

impl KVMap for DashMap {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for DashMap {
    fn set(&mut self, key: &[u8], value: &[u8]) {
        self.0.insert(key.into(), value.into());
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        match self.0.get(key) {
            Some(v) => Some(v.clone()),
            None => None,
        }
    }
}

inventory::submit! {
    Registry::new("dashmap", DashMap::new_benchkvmap)
}

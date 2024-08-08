//! Adapter implementation of [`std::collections::BTreeMap`].
//!
//! ## Configuration Format
//!
//! ### [`Mutex`]-based:
//!
//! ``` toml
//! [map]
//! name = "mutex_btreemap"
//! ```
//!
//! This store is [`KVMap`].
//!
//! ### [`RwLock`]-based:
//! ``` toml
//! [map]
//! name = "rwlock_btreemap"
//! ```
//!
//! This store is [`KVMap`].

use crate::stores::*;
use parking_lot::{Mutex, RwLock};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct MutexBTreeMap(Arc<Mutex<BTreeMap<Box<[u8]>, Box<[u8]>>>>);

impl MutexBTreeMap {
    pub fn new() -> Self {
        Self(Arc::new(
            Mutex::new(BTreeMap::<Box<[u8]>, Box<[u8]>>::new()),
        ))
    }

    pub fn new_benchkvmap(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Regular(Arc::new(Box::new(Self::new())))
    }
}

impl KVMap for MutexBTreeMap {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for MutexBTreeMap {
    fn set(&mut self, key: &[u8], value: &[u8]) {
        self.0.lock().insert(key.into(), value.into());
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        match self.0.lock().get(key) {
            Some(v) => Some(v.clone()),
            None => None,
        }
    }

    fn delete(&mut self, key: &[u8]) {
        self.0.lock().remove(key);
    }

    fn scan(&mut self, _key: &[u8], _n: usize) -> Vec<(Box<[u8]>, Box<[u8]>)> {
        // technically iteration is supported but querying a specific range is not a stable feature
        unimplemented!("Range query is not supported");
    }
}

inventory::submit! {
    Registry::new("mutex_btreemap", MutexBTreeMap::new_benchkvmap)
}

#[derive(Clone)]
pub struct RwLockBTreeMap(Arc<RwLock<BTreeMap<Box<[u8]>, Box<[u8]>>>>);

impl RwLockBTreeMap {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(
            BTreeMap::<Box<[u8]>, Box<[u8]>>::new(),
        )))
    }

    pub fn new_benchkvmap(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Regular(Arc::new(Box::new(Self::new())))
    }
}

impl KVMap for RwLockBTreeMap {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for RwLockBTreeMap {
    fn set(&mut self, key: &[u8], value: &[u8]) {
        self.0.write().insert(key.into(), value.into());
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        match self.0.read().get(key) {
            Some(v) => Some(v.clone()),
            None => None,
        }
    }

    fn delete(&mut self, key: &[u8]) {
        self.0.write().remove(key);
    }

    fn scan(&mut self, _key: &[u8], _n: usize) -> Vec<(Box<[u8]>, Box<[u8]>)> {
        // technically iteration is supported but querying a specific range is not a stable feature
        unimplemented!("Range query is not supported");
    }
}

inventory::submit! {
    Registry::new("rwlock_btreemap", RwLockBTreeMap::new_benchkvmap)
}

//! Adapter implementation of [`hashbrown::HashMap`]. Internally sharded.
//!
//! ## Configuration Format
//!
//! ### [`Mutex`]-based:
//!
//! ``` toml
//! [map]
//! name = "mutex_hashmap"
//! shards = ... # number of shards
//! ```
//!
//! This store is [`KVMap`].
//!
//! ### [`RwLock`]-based:
//! ``` toml
//! [map]
//! name = "rwlock_hashmap"
//! shards = ... # number of shards
//! ```
//!
//! This store is [`KVMap`].

use crate::stores::{BenchKVMap, Registry};
use crate::*;
use ::hashbrown::HashMap;
use ahash::AHasher;
use parking_lot::{Mutex, RwLock};
use serde::Deserialize;
use std::hash::Hasher;
use std::sync::Arc;

/// Calculate the [`u64`] hash value of a given key using [`AHasher`].
pub fn hash(key: &[u8]) -> u64 {
    let mut hasher = AHasher::default();
    hasher.write(key);
    u64::from(hasher.finish())
}

pub fn shard(key: &[u8], nr_shards: usize) -> usize {
    let hash = hash(key);
    usize::try_from(hash).unwrap() % nr_shards
}

/// A wrapper around raw [`HashMap`] with variable-sized keys and values.
///
/// It is used as the building block of other types. Note that this is not [`KVMap`].
pub type BaseHashMap = HashMap<Box<[u8]>, Box<[u8]>>;

#[derive(Clone)]
pub struct MutexHashMap {
    nr_shards: usize,
    shards: Arc<Vec<Mutex<BaseHashMap>>>,
}

#[derive(Deserialize)]
pub struct MutexHashMapOpt {
    pub shards: usize,
}

impl MutexHashMap {
    pub fn new(opt: &MutexHashMapOpt) -> Self {
        let nr_shards = opt.shards;
        let mut shards = Vec::<Mutex<BaseHashMap>>::with_capacity(nr_shards);
        for _ in 0..nr_shards {
            shards.push(Mutex::new(BaseHashMap::new()));
        }
        let shards = Arc::new(shards);
        Self { nr_shards, shards }
    }

    pub fn new_benchkvmap(opt: &toml::Table) -> BenchKVMap {
        let opt: MutexHashMapOpt = opt.clone().try_into().unwrap();
        BenchKVMap::Regular(Box::new(Self::new(&opt)))
    }
}

impl KVMap for MutexHashMap {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for MutexHashMap {
    fn set(&mut self, key: &[u8], value: &[u8]) {
        let sid = shard(key, self.nr_shards);
        self.shards[sid].lock().insert(key.into(), value.into());
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        let sid = shard(key, self.nr_shards);
        match self.shards[sid].lock().get(key) {
            Some(v) => Some(v.clone()),
            None => None,
        }
    }

    fn delete(&mut self, key: &[u8]) {
        let sid = shard(key, self.nr_shards);
        self.shards[sid].lock().remove(key);
    }

    fn scan(&mut self, _key: &[u8], _n: usize) -> Vec<(Box<[u8]>, Box<[u8]>)> {
        unimplemented!("Range query is not supported");
    }
}

inventory::submit! {
    Registry::new("mutex_hashmap", MutexHashMap::new_benchkvmap)
}

// }}} mutex_hashmap

// {{{ rwlock_hashmap

#[derive(Clone)]
pub struct RwLockHashMap {
    pub nr_shards: usize,
    shards: Arc<Vec<RwLock<BaseHashMap>>>,
}

#[derive(Deserialize)]
pub struct RwLockHashMapOpt {
    pub shards: usize,
}

impl RwLockHashMap {
    pub fn new(opt: &RwLockHashMapOpt) -> Self {
        let nr_shards = opt.shards;
        let mut shards = Vec::<RwLock<BaseHashMap>>::with_capacity(nr_shards);
        for _ in 0..nr_shards {
            shards.push(RwLock::new(BaseHashMap::new()));
        }
        let shards = Arc::new(shards);
        Self { nr_shards, shards }
    }

    pub fn new_benchkvmap(opt: &toml::Table) -> BenchKVMap {
        let opt: RwLockHashMapOpt = opt.clone().try_into().unwrap();
        BenchKVMap::Regular(Box::new(Self::new(&opt)))
    }
}

impl KVMap for RwLockHashMap {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for RwLockHashMap {
    fn set(&mut self, key: &[u8], value: &[u8]) {
        let sid = shard(key, self.nr_shards);
        self.shards[sid].write().insert(key.into(), value.into());
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        let sid = shard(key, self.nr_shards);
        match self.shards[sid].read().get(key) {
            Some(v) => Some(v.clone()),
            None => None,
        }
    }

    fn delete(&mut self, key: &[u8]) {
        let sid = shard(key, self.nr_shards);
        self.shards[sid].write().remove(key);
    }

    fn scan(&mut self, _key: &[u8], _n: usize) -> Vec<(Box<[u8]>, Box<[u8]>)> {
        unimplemented!("Range query is not supported");
    }
}

inventory::submit! {
    Registry::new("rwlock_hashmap", RwLockHashMap::new_benchkvmap)
}

// }}} rwlock_hashmap

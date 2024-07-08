use crate::bench::{BenchKVMap, Registry};
use crate::client::KVClient;
use crate::*;
use ahash::AHasher;
use hashbrown::HashMap;
use parking_lot::{Mutex, RwLock};
use serde::Deserialize;
use std::hash::Hasher;
use std::rc::Rc;
use std::sync::Arc;

pub fn hash(key: &[u8]) -> u64 {
    let mut hasher = AHasher::default();
    hasher.write(key);
    u64::from(hasher.finish())
}

pub fn find_shard(key: &[u8], nr_shards: usize) -> usize {
    let mut hasher = AHasher::default();
    hasher.write(key);
    let hash = u64::from(hasher.finish());
    usize::try_from(hash).unwrap() % nr_shards
}

/// BaseHashMap is a wrapper around raw HashMap with variable-sized keys and values.
/// It is used as the builing block of other types, like a shard, or a property
/// in the delegation setup. Note that BaseHashMap is not KVMap.
pub type BaseHashMap = HashMap<Box<[u8]>, Box<[u8]>>;

// {{{ nullmap

/// NullMap does nothing. It can be used to measure overheads in the future.
#[derive(Clone)]
pub struct NullMap;

impl NullMap {
    pub fn new() -> Self {
        Self
    }

    pub fn new_benchkvmap(_opt: &toml::Table) -> BenchKVMap {
        BenchKVMap::Regular(Box::new(Self::new()))
    }
}

impl KVMap for NullMap {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for NullMap {
    fn set(&mut self, _key: &[u8], _value: &[u8]) {}

    fn get(&mut self, _key: &[u8]) -> Option<Box<[u8]>> {
        None
    }
}

inventory::submit! {
    Registry::new("nullmap", NullMap::new_benchkvmap)
}

// }}} nullmap

// {{{ mutex_hashmap

/// MutexHashMap is a locking-based KV. It uses Mutex<BaseHashMap> for sharding to
/// avoid a monolithic lock. It implements KVMap, nartually.
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
        let sid = find_shard(key, self.nr_shards);
        self.shards[sid].lock().insert(key.into(), value.into());
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        let sid = find_shard(key, self.nr_shards);
        match self.shards[sid].lock().get(key) {
            Some(v) => Some(v.clone()),
            None => None,
        }
    }
}

inventory::submit! {
    Registry::new("mutex_hashmap", MutexHashMap::new_benchkvmap)
}

// }}} mutex_hashmap

// {{{ rwlock_hashmap

/// RwLockHashMap similar to MutexHashMap but it uses RwLocks instead of
/// Mutex.
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
        let sid = find_shard(key, self.nr_shards);
        self.shards[sid].write().insert(key.into(), value.into());
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        let sid = find_shard(key, self.nr_shards);
        match self.shards[sid].read().get(key) {
            Some(v) => Some(v.clone()),
            None => None,
        }
    }
}

inventory::submit! {
    Registry::new("rwlock_hashmap", RwLockHashMap::new_benchkvmap)
}

// }}} rwlock_hashmap

// {{{ dashmap

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

// }}} dashmap

// {{{ remotemap

pub struct RemoteMap {
    host: String,
    port: String,
}

pub struct RemoteMapHandle {
    client: KVClient,
    responder: Rc<dyn AsyncResponder>,
}

#[derive(Deserialize)]
pub struct RemoteMapOpt {
    host: String,
    port: String,
}

impl RemoteMap {
    pub fn new(opt: &RemoteMapOpt) -> Self {
        Self {
            host: opt.host.clone(),
            port: opt.port.clone(),
        }
    }

    pub fn new_benchkvmap(opt: &toml::Table) -> BenchKVMap {
        let opt: RemoteMapOpt = opt.clone().try_into().unwrap();
        BenchKVMap::Async(Box::new(Self::new(&opt)))
    }
}

impl AsyncKVMap for RemoteMap {
    fn handle(&self, responder: Rc<dyn AsyncResponder>) -> Box<dyn AsyncKVMapHandle> {
        Box::new(RemoteMapHandle {
            client: KVClient::new(&self.host, &self.port).unwrap(),
            responder,
        })
    }
}

impl AsyncKVMapHandle for RemoteMapHandle {
    fn submit(&mut self, requests: &Vec<Request>) {
        self.client.send_requests(requests);
    }

    fn drain(&mut self) {
        for r in self.client.recv_responses().into_iter() {
            self.responder.callback(r);
        }
    }
}

inventory::submit! {
    Registry::new("remotemap", RemoteMap::new_benchkvmap)
}

// }}} remotemap

#[cfg(test)]
mod tests {
    use super::*;

    fn map_test(map: &impl KVMap) {
        let mut handle = map.handle();
        // insert + get
        handle.set(b"foo", b"bar");
        assert_eq!(handle.get(b"foo"), Some((*b"bar").into()));
        assert_eq!(handle.get(b"f00"), None);

        // update
        handle.set(b"foo", b"0ar");
        assert_eq!(handle.get(b"foo"), Some((*b"0ar").into()));
    }

    #[test]
    fn nullmap() {
        let mut map = NullMap::new();
        assert!(map.get("foo".as_bytes().into()).is_none());
    }

    #[test]
    fn mutex_hashmap() {
        let opt = MutexHashMapOpt { shards: 512 };
        let mut map = MutexHashMap::new(&opt);
        map_test(&mut map);
    }

    #[test]
    fn rwlock_hashmap() {
        let opt = RwLockHashMapOpt { shards: 512 };
        let mut map = RwLockHashMap::new(&opt);
        map_test(&mut map);
    }

    #[test]
    fn dashmap() {
        let mut map = DashMap::new();
        map_test(&mut map);
    }
}

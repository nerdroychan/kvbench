//! The implementation of built-in key-value stores, and some helper functions.
//!
//! ## Configuration Format
//!
//! The configuration of a key-value store is stored in a dictionary named `map`. Therefore, a
//! store's configuration file looks like the following:
//!
//! ```toml
//! [map]
//! name = "..."
//! # option1 = ...
//! # option2 = ...
//!
//! ...
//! ```
//! The field `name` must be given and it should be equal to the name registered by the store.
//! Other than `name`, all the fields are parsed as a string map and will be hand over to the
//! constructor of the store's constructor function. For available options other than `name`, one
//! can refer to the module-level documentation of a specific store.
//!
//! ## Registering New Stores
//!
//! When users would like to dynamically register new key-value stores from their own crate, first
//! of all, they need to implement the corresponding [`KVMap`]/[`KVMapHandle`]
//! (or [`AsyncKVMap`]/[`AsyncKVMapHandle`]) for the store. Then, they need to create a constructor
//! function with a signature of `fn(&toml::Table) -> BenchKVMap`.
//!
//! The final step is to register the store's constructor (along with its name) using
//! [`inventory`]. A minimal example would be: `inventory::submit! { Registry::new("name",
//! constructor_fn) };`.
//!
//! The source code of all built-in stores provide good examples on this process.

use crate::bench::Benchmark;
use crate::*;
use ahash::AHasher;
use hashbrown::HashMap;
use log::debug;
use std::hash::Hasher;
use toml::Table;

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

/// A unified enum for a created key-value store that is ready to run.
pub enum BenchKVMap {
    Regular(Box<dyn KVMap>),
    Async(Box<dyn AsyncKVMap>),
}

impl BenchKVMap {
    /// Wraps the real `bench` function of the store.
    pub fn bench(self, phases: &Vec<Arc<Benchmark>>) {
        match self {
            BenchKVMap::Regular(map) => {
                KVMap::bench(map, phases);
            }
            BenchKVMap::Async(map) => {
                AsyncKVMap::bench(map, phases);
            }
        };
    }
}

/// The centralized registry that maps the name of newly added key-value store to its constructor
/// function.
///
/// A user-defined store can use the [`inventory::submit!`] macro to register their own stores to
/// be used in the benchmark framework.
pub struct Registry<'a> {
    pub(crate) name: &'a str,
    constructor: fn(&Table) -> BenchKVMap,
}

impl<'a> Registry<'a> {
    pub const fn new(name: &'a str, constructor: fn(&Table) -> BenchKVMap) -> Self {
        Self { name, constructor }
    }
}

inventory::collect!(Registry<'static>);

/// An aggregated option enum that can be parsed from a TOML string. It contains all necessary
/// parameters for each type of maps to be created.
#[derive(Deserialize, Clone, Debug)]
pub(crate) struct BenchKVMapOpt {
    name: String,
    #[serde(flatten)]
    opt: Table,
}

impl BenchKVMap {
    pub(crate) fn new(opt: &BenchKVMapOpt) -> BenchKVMap {
        // construct the hashmap.. this will be done every time
        let mut registered: HashMap<&'static str, fn(&Table) -> BenchKVMap> = HashMap::new();
        for r in inventory::iter::<Registry> {
            debug!("Adding supported kvmap: {}", r.name);
            assert!(registered.insert(r.name, r.constructor).is_none()); // no existing name
        }
        let f = registered.get(opt.name.as_str()).unwrap_or_else(|| {
            panic!("map {} not found in registry", opt.name);
        });
        f(&opt.opt)
    }
}

pub mod btreemap;
pub mod chashmap;
pub mod contrie;
pub mod dashmap;
pub mod flurry;
pub mod hashmap;
pub mod null;
pub mod papaya;
pub mod remote;
pub mod scc;

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

        // delete
        handle.delete(b"foo");
        assert_eq!(handle.get(b"foo"), None);
    }

    #[test]
    fn nullmap() {
        let mut map = null::NullMap::new();
        assert!(map.get("foo".as_bytes().into()).is_none());
    }

    #[test]
    fn mutex_hashmap() {
        let opt = hashmap::MutexHashMapOpt { shards: 512 };
        let mut map = hashmap::MutexHashMap::new(&opt);
        map_test(&mut map);
    }

    #[test]
    fn rwlock_hashmap() {
        let opt = hashmap::RwLockHashMapOpt { shards: 512 };
        let mut map = hashmap::RwLockHashMap::new(&opt);
        map_test(&mut map);
    }

    #[test]
    fn dashmap() {
        let mut map = dashmap::DashMap::new();
        map_test(&mut map);
    }

    #[test]
    fn contrie() {
        let mut map = contrie::Contrie::new();
        map_test(&mut map);
    }

    #[test]
    fn chashmap() {
        let mut map = chashmap::CHashMap::new();
        map_test(&mut map);
    }

    #[test]
    fn scchashmap() {
        let mut map = scc::SccHashMap::new();
        map_test(&mut map);
    }

    #[test]
    fn flurry() {
        let mut map = flurry::Flurry::new();
        map_test(&mut map);
    }

    #[test]
    fn papaya() {
        let mut map = papaya::Papaya::new();
        map_test(&mut map);
    }

    #[test]
    fn mutex_btreemap() {
        let mut map = btreemap::MutexBTreeMap::new();
        map_test(&mut map);
    }

    #[test]
    fn rwlock_btreemap() {
        let mut map = btreemap::RwLockBTreeMap::new();
        map_test(&mut map);
    }
}

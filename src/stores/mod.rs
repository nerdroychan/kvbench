//! The implementation of built-in key-value stores, and some util functions.
//!
//! ## Configuration Format
//!
//! The configuration of a key-value store is stored in a dictionary named `map`. Therefore, a
//! store's config file looks like the following:
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
//! Other than `name`, all the fileds are parsed as a string map and will be hand over to the
//! constructor of the store's constructor function.
//!
//! ## Registering New Stores
//!
//! When users would like to dynamically register new key-value stores from their own crate, first
//! of all, they need to implemement the corresponding [`KVMap`]/[`KVMapHandle`]
//! (or [`AsyncKVMap`]/[`AsyncKVMapHandle`]) for the store. Then, they need to create a construcor
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

/// An aggregated option enum that can be parsed from a toml string. It contains all necessary
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

pub mod chashmap;
pub mod contrie;
pub mod dashmap;
pub mod flurry;
pub mod hashmap;
pub mod null;
pub mod papaya;
pub mod remote;
pub mod scc;

pub use chashmap::*;
pub use contrie::*;
pub use dashmap::*;
pub use flurry::*;
pub use hashmap::*;
pub use null::*;
pub use papaya::*;
pub use remote::*;
pub use scc::*;

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

    #[test]
    fn contrie() {
        let mut map = Contrie::new();
        map_test(&mut map);
    }

    #[test]
    fn chashmap() {
        let mut map = CHashMap::new();
        map_test(&mut map);
    }

    #[test]
    fn scchashmap() {
        let mut map = SccHashMap::new();
        map_test(&mut map);
    }

    #[test]
    fn flurry() {
        let mut map = Flurry::new();
        map_test(&mut map);
    }

    #[test]
    fn papaya() {
        let mut map = Papaya::new();
        map_test(&mut map);
    }
}

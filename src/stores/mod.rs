use ahash::AHasher;
use std::hash::Hasher;

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
    use crate::*;

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

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

pub mod dashmap;
pub mod hashmap;
pub mod null;
pub mod remote;

pub use dashmap::*;
pub use hashmap::*;
pub use null::*;
pub use remote::*;

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

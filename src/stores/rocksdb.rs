//! Adapter implementation of [`rocksdb`].
//!
//! ## Configuration Format
//!
//! ``` toml
//! [map]
//! name = "rocksdb"
//! path = "..." # path to the rocksdb data directory
//! ```
//!
//! This store is [`KVMap`].

use crate::stores::{BenchKVMap, Registry};
use crate::*;
use rocksdb::{Direction, IteratorMode, DB};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct RocksDBOpt {
    pub path: String,
}

#[derive(Clone)]
pub struct RocksDB {
    db: Arc<DB>,
}

impl RocksDB {
    pub fn new(opt: &RocksDBOpt) -> Self {
        let db = Arc::new(DB::open_default(&opt.path).unwrap());
        Self { db }
    }

    pub fn new_benchkvmap(opt: &toml::Table) -> BenchKVMap {
        let opt: RocksDBOpt = opt.clone().try_into().unwrap();
        BenchKVMap::Regular(Box::new(Self::new(&opt)))
    }
}

impl KVMap for RocksDB {
    fn handle(&self) -> Box<dyn KVMapHandle> {
        Box::new(self.clone())
    }
}

impl KVMapHandle for RocksDB {
    fn set(&mut self, key: &[u8], value: &[u8]) {
        assert!(self.db.put(key, value).is_ok());
    }

    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        if let Ok(v) = self.db.get(key) {
            v.map(|vec| vec.into_boxed_slice())
        } else {
            None
        }
    }

    fn delete(&mut self, key: &[u8]) {
        assert!(self.db.delete(key).is_ok());
    }

    fn scan(&mut self, key: &[u8], n: usize) -> Vec<Box<[u8]>> {
        let mut kv = Vec::with_capacity(n);
        let iter = self
            .db
            .iterator(IteratorMode::From(key, Direction::Forward));
        let mut i = 0;
        for item in iter {
            if i == n {
                break;
            }
            kv.push(item.unwrap().1);
            i += 1;
        }
        kv
    }
}

inventory::submit! {
    Registry::new("rocksdb", RocksDB::new_benchkvmap)
}

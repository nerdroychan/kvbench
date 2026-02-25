//! A client of a client-side sharded key-value server. Each endpoint points to an independent
//! store. But the client will shard the keys internally and send requests to the correct shard.
//! The server can be backed by any stores available. However, the shading method is hash-based by
//! default. Therefore, range queries are not supported.
//!
//! ## Configuration Format
//!
//! ``` toml
//! [map]
//! name = "remotesharded"
//!
//! [[map.addr]]
//! host = "..." # host 1
//! port = "..." # port 1
//!
//! [[map.addr]]
//! host = "..." # host 2
//! port = "..." # port 2
//! ```
//!
//! This store is [`AsyncKVMap`].

use crate::stores::remote::{RemoteMap, RemoteMapOpt};
use crate::stores::{BenchKVMap, Registry};
use crate::*;
use djb_hash::x33a::X33a;
use serde::Deserialize;
use std::hash::Hasher;
use std::rc::Rc;

/// The hash function used by remote sharded map, base on djbhash.
///
/// Note that we do not use the same hash function as the existing hash maps, because if the number
/// of shards in the client and number of shard in the server has a common denominator, it will
/// cause certain shards not being used at all.
fn hash(key: &[u8]) -> u64 {
    let mut hasher = X33a::new();
    hasher.write(key);
    return hasher.finish();
}

fn shard(key: &[u8], nr_shards: usize) -> usize {
    let hash = hash(key);
    usize::try_from(hash).unwrap() % nr_shards
}

pub struct RemoteShardedMap {
    maps: Vec<RemoteMap>,
}

pub struct RemoteShardedMapHandle(Vec<Box<dyn AsyncKVMapHandle>>);

#[derive(Deserialize)]
pub struct RemoteShardedMapOpt {
    addr: Vec<RemoteMapOpt>,
}

impl RemoteShardedMap {
    pub fn new(opt: &RemoteShardedMapOpt) -> Self {
        Self {
            maps: opt.addr.iter().map(|a| RemoteMap::new(a)).collect(),
        }
    }

    pub fn new_benchkvmap(opt: &toml::Table) -> BenchKVMap {
        let opt: RemoteShardedMapOpt = opt.clone().try_into().unwrap();
        BenchKVMap::Async(Arc::new(Box::new(Self::new(&opt))))
    }
}

impl AsyncKVMap for RemoteShardedMap {
    fn handle(&self, responder: Rc<dyn AsyncResponder>) -> Box<dyn AsyncKVMapHandle> {
        let nr_shards = self.maps.len();
        let handles = (0..nr_shards)
            .into_iter()
            .map(|i| self.maps[i].handle(responder.clone()))
            .collect();
        Box::new(RemoteShardedMapHandle(handles))
    }
}

fn shard_requests(requests: &Vec<Request>, nr_shards: usize) -> Vec<Vec<Request>> {
    let mut ret: Vec<Vec<Request>> = (0..nr_shards)
        .into_iter()
        .map(|_| Vec::<Request>::new())
        .collect();
    for r in requests.iter() {
        match &r.op {
            Operation::Set { key, value: _ } => ret[shard(key, nr_shards)].push(r.clone()),
            Operation::Get { key } => ret[shard(key, nr_shards)].push(r.clone()),
            Operation::Delete { key } => ret[shard(key, nr_shards)].push(r.clone()),
            Operation::Scan { key: _, n: _ } => {
                unimplemented!("remotesharded doesn't support range query")
            }
        }
    }
    ret
}

impl AsyncKVMapHandle for RemoteShardedMapHandle {
    fn submit(&mut self, requests: &Vec<Request>) {
        let sharded_requests = shard_requests(requests, self.0.len());
        for (shard, req) in sharded_requests.iter().enumerate() {
            self.0[shard].submit(req);
        }
    }

    fn drain(&mut self) {
        for r in self.0.iter_mut() {
            r.drain();
        }
    }
}

inventory::submit! {
    Registry::new("remoteshardedmap", RemoteShardedMap::new_benchkvmap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        let opt: RemoteShardedMapOpt = toml::from_str(
            r#"
            [[addr]]
            host = "127.0.0.1"
            port = "8080"

            [[addr]]
            host = "127.0.0.1"
            port = "8081"
            "#,
        )
        .unwrap();
        let map = RemoteShardedMap::new(&opt);
        assert_eq!(map.maps.len(), 2);
    }

    #[test]
    #[should_panic(expected = "not implemented")]
    fn shard_requests_invalid() {
        let mut requests = Vec::new();
        requests.push(Request {
            id: 0,
            op: Operation::Scan {
                key: Box::new([0u8; 8]),
                n: 2,
            },
        });
        let _ = super::shard_requests(&requests, 10);
    }

    #[test]
    fn shard_requests() {
        let mut requests = Vec::new();
        for i in 0..1000 {
            requests.push(Request {
                id: i,
                op: Operation::Get {
                    key: Box::new(i.to_be_bytes()),
                },
            });
            requests.push(Request {
                id: i * 2 + 1,
                op: Operation::Set {
                    key: Box::new(i.to_be_bytes()),
                    value: Box::new([0u8; 16]),
                },
            });
        }
        let sharded_requests = super::shard_requests(&requests, 10);
        assert_eq!(sharded_requests.len(), 10);
        let mut count = 0;
        for r in sharded_requests.iter() {
            count += r.len();
        }
        assert_eq!(count, 2000);
    }
}

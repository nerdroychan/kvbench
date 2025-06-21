//! A client of a gateway-replicated key-value server. Each endpoint points to the same store.
//! The server can be backed by any stores available.
//!
//! ## Configuration Format
//!
//! ``` toml
//! [map]
//! name = "remotereplicated"
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

use crate::server::KVClient;
use crate::stores::remote::{RemoteMap, RemoteMapOpt};
use crate::stores::{BenchKVMap, Registry};
use crate::*;
use serde::Deserialize;
use std::rc::Rc;
use std::sync::atomic::AtomicUsize;

pub struct RemoteReplicatedMap {
    maps: Vec<RemoteMap>,
    next: AtomicUsize,
}

pub struct RemoteReplicatedMapHandle {
    client: KVClient,
    responder: Rc<dyn AsyncResponder>,
}

#[derive(Deserialize)]
pub struct RemoteReplicatedMapOpt {
    addr: Vec<RemoteMapOpt>,
}

impl RemoteReplicatedMap {
    pub fn new(opt: &RemoteReplicatedMapOpt) -> Self {
        Self {
            maps: opt.addr.iter().map(|a| RemoteMap::new(a)).collect(),
            next: AtomicUsize::new(0),
        }
    }

    pub fn new_benchkvmap(opt: &toml::Table) -> BenchKVMap {
        let opt: RemoteReplicatedMapOpt = opt.clone().try_into().unwrap();
        BenchKVMap::Async(Arc::new(Box::new(Self::new(&opt))))
    }
}

impl AsyncKVMap for RemoteReplicatedMap {
    fn handle(&self, responder: Rc<dyn AsyncResponder>) -> Box<dyn AsyncKVMapHandle> {
        let next = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let map = &self.maps[next % self.maps.len()];
        map.handle(responder)
    }
}

impl AsyncKVMapHandle for RemoteReplicatedMapHandle {
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
    Registry::new("remotereplicatedmap", RemoteReplicatedMap::new_benchkvmap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse() {
        let opt: RemoteReplicatedMapOpt = toml::from_str(
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
        let map = RemoteReplicatedMap::new(&opt);
        assert_eq!(map.maps.len(), 2);
    }
}

//! A client of a key-value server. The server can be backed by any stores available.
//!
//! ## Configuration Format
//!
//! ``` toml
//! [map]
//! name = "remote"
//! host = "..." # hostname of the server
//! port = "..." # port of the server
//! ```
//!
//! This store is [`AsyncKVMap`].

use crate::server::KVClient;
use crate::stores::{BenchKVMap, Registry};
use crate::*;
use serde::Deserialize;
use std::rc::Rc;

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
        BenchKVMap::Async(Arc::new(Box::new(Self::new(&opt))))
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

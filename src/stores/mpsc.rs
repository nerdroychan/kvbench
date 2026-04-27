//! MPSC-based, sharded hashmap.

use crate::stores::{BenchKVMap, Registry};
use crate::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

type Key = Box<[u8]>;
type Value = Box<[u8]>;

struct WorkerRequest {
    req: Request,
    reply_tx: Sender<Response>,
}

struct Shard {
    tx: Sender<WorkerRequest>,
    _handle: JoinHandle<()>,
}

pub struct MpscHashMap {
    shards: Vec<Shard>,
}

#[derive(Deserialize)]
pub struct MpscHashMapOpt {
    pub shards: usize,
}

impl MpscHashMap {
    pub fn new(opt: &MpscHashMapOpt) -> Self {
        let nr_shards = opt.shards.max(1);
        let mut shards = Vec::with_capacity(nr_shards);

        for _ in 0..nr_shards {
            let (tx, rx) = mpsc::channel::<WorkerRequest>();

            let handle = thread::spawn(move || {
                let mut store: HashMap<Key, Value> = HashMap::new();

                for WorkerRequest { req, reply_tx } in rx {
                    let response = match req.op {
                        Operation::Get { key } => {
                            let data = store.get(&key).cloned().map(|v| vec![v]);
                            Response { id: req.id, data }
                        }
                        Operation::Set { key, value } => {
                            store.insert(key, value);
                            Response {
                                id: req.id,
                                data: None,
                            }
                        }
                        Operation::Delete { key } => {
                            let _ = store.remove(&key);
                            Response {
                                id: req.id,
                                data: None,
                            }
                        }
                        Operation::Scan { .. } => {
                            // Scan not supported in this simple sharded impl (same as remotesharded)
                            Response {
                                id: req.id,
                                data: None,
                            }
                        }
                    };

                    let _ = reply_tx.send(response);
                }
            });

            shards.push(Shard {
                tx,
                _handle: handle,
            });
        }

        Self { shards }
    }

    pub fn new_benchkvmap(opt: &toml::Table) -> BenchKVMap {
        let opt: MpscHashMapOpt = opt.clone().try_into().unwrap();
        BenchKVMap::Async(Arc::new(Box::new(Self::new(&opt))))
    }
}

impl AsyncKVMap for MpscHashMap {
    fn handle(&self, responder: Rc<dyn AsyncResponder>) -> Box<dyn AsyncKVMapHandle> {
        // === ONE response channel per client (as you requested) ===
        let (reply_tx, reply_rx) = mpsc::channel::<Response>();

        Box::new(MpscShardedHandle {
            shards: self.shards.iter().map(|s| s.tx.clone()).collect(),
            reply_tx,
            reply_rx,
            responder,
            pending: 0,
        })
    }
}

struct MpscShardedHandle {
    shards: Vec<Sender<WorkerRequest>>,
    reply_tx: Sender<Response>,
    reply_rx: Receiver<Response>,
    responder: Rc<dyn AsyncResponder>,
    pending: usize,
}

fn shard_key(key: &[u8], nr_shards: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    (hasher.finish() as usize) % nr_shards
}

impl AsyncKVMapHandle for MpscShardedHandle {
    fn submit(&mut self, requests: &Vec<Request>) {
        let nr_shards = self.shards.len();
        for req in requests {
            let shard_idx = match &req.op {
                Operation::Get { key } | Operation::Set { key, .. } | Operation::Delete { key } => {
                    shard_key(key, nr_shards)
                }
                Operation::Scan { .. } => 0,
            };

            let worker_req = WorkerRequest {
                req: req.clone(),
                reply_tx: self.reply_tx.clone(), // send the client's reply channel to the shard
            };

            let _ = self.shards[shard_idx].send(worker_req);
        }
        self.pending += requests.len();
    }

    fn drain(&mut self) {
        // Non-blocking drain: pull as many responses as available from our single client channel
        while self.pending > 0 {
            match self.reply_rx.try_recv() {
                Ok(response) => {
                    self.responder.callback(response);
                    self.pending -= 1;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
    }
}

inventory::submit! {
    Registry::new("mpsc_hashmap", MpscHashMap::new_benchkvmap)
}

//! A key-value server/client implementation.

use crate::stores::{BenchKVMap, BenchKVMapOpt};
use crate::thread::{JoinHandle, Thread};
use crate::*;
use figment::providers::{Env, Format, Toml};
use figment::Figment;
use hashbrown::HashMap;
use log::debug;
use mio::net::TcpStream;
use mio::{Events, Interest, Poll, Token};
use std::cell::RefCell;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream as StdTcpStream};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

/// Requests are sent in a batch. Here we do not send the requests in the batch one by one, but
/// instead in a vector, because the server implementation uses event-based notification. It does
/// not know how many requests are there during an event, so it may not read all requests,
/// especially when the batch is large.
fn write_requests(writer: &mut impl Write, requests: &Vec<Request>) -> Result<(), bincode::Error> {
    bincode::serialize_into(writer, requests)
}

fn read_requests(reader: &mut impl Read) -> Result<Vec<Request>, bincode::Error> {
    bincode::deserialize_from::<_, Vec<Request>>(reader)
}

/// Responses have a customized header and the (de)serialization process is manual because the
/// payload (data) in a response may be from a reference. It is preferable to directly write the
/// bytes from the reference to the writer instead of creating a new [`Response`] and perform a
/// copy of the payload data.
#[derive(Serialize, Deserialize)]
struct ResponseHeader {
    id: usize,
    len: usize,
}

fn write_response(
    writer: &mut impl Write,
    id: usize,
    data: Option<&[u8]>,
) -> Result<(), bincode::Error> {
    let len = match data {
        Some(data) => data.len(),
        None => 0,
    };
    let header = ResponseHeader { id, len };
    if let Err(e) = bincode::serialize_into(&mut *writer, &header) {
        return Err(e);
    }
    // has payload
    if len != 0 {
        if let Err(e) = writer.write_all(data.unwrap()) {
            return Err(bincode::Error::from(e));
        }
    }
    Ok(())
}

fn read_response(reader: &mut impl Read) -> Result<Response, bincode::Error> {
    let header = bincode::deserialize_from::<_, ResponseHeader>(&mut *reader)?;
    let id = header.id;
    let len = header.len;
    if len != 0 {
        let mut data = vec![0u8; len].into_boxed_slice();
        if let Err(e) = reader.read_exact(&mut data[..]) {
            Err(bincode::Error::from(e))
        } else {
            Ok(Response {
                id,
                data: Some(data),
            })
        }
    } else {
        Ok(Response { id, data: None })
    }
}

const POLLING_TIMEOUT: Option<Duration> = Some(Duration::new(0, 0));

enum WorkerMsg {
    NewConnection(StdTcpStream, SocketAddr),
    Terminate,
}

fn serve_requests_regular(
    handle: &mut Box<dyn KVMapHandle>,
    requests: &Vec<Request>,
    writer: &mut ResponseWriter,
) {
    for Request { id, op: body } in requests.iter() {
        let id = id.clone();
        match body {
            Operation::Set { ref key, ref value } => {
                handle.set(key, value);
                assert!(write_response(&mut *writer, id, None).is_ok());
            }
            Operation::Get { ref key } => match handle.get(key) {
                Some(v) => {
                    assert!(write_response(&mut *writer, id, Some(&v[..]),).is_ok());
                }
                None => {
                    assert!(write_response(&mut *writer, id, None).is_ok());
                }
            },
            Operation::Delete { ref key } => {
                handle.delete(key);
                assert!(write_response(&mut *writer, id, None).is_ok());
            }
        }
    }
}

pub(crate) fn serve_requests_async(
    handle: &mut Box<dyn AsyncKVMapHandle>,
    requests: &Vec<Request>,
) {
    handle.submit(requests);
}

/// Wrapper around [`TcpStream`] to enable multi-ownership in reader/writer for the same connection
#[derive(Clone)]
struct RcTcpStream(Rc<TcpStream>);

impl RcTcpStream {
    fn new(stream: TcpStream) -> RcTcpStream {
        RcTcpStream(Rc::new(stream))
    }
}

impl Read for RcTcpStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        (&*self.0).read(buf)
    }
}

impl Write for RcTcpStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        (&*self.0).write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        (&*self.0).flush()
    }
}

/// Wrapper around buffered request reader. It is always read only by the corresponding worker
/// thread, so there is no sharing.
struct RequestReader(BufReader<RcTcpStream>);

impl RequestReader {
    fn new(stream: RcTcpStream) -> Self {
        Self(BufReader::new(stream))
    }
}

impl Read for RequestReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

impl BufRead for RequestReader {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.0.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        self.0.consume(amt);
    }
}

/// Unlike reader, writer is registered to one or more handles. Therefore, it needs a cell to work,
/// and potentially a wrapper [`Rc`].
struct ResponseWriter(RefCell<BufWriter<RcTcpStream>>);

impl ResponseWriter {
    fn new(stream: RcTcpStream) -> Self {
        Self(RefCell::new(BufWriter::new(stream)))
    }
}

impl Write for ResponseWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.borrow_mut().flush()
    }
}

enum WriterOrHandle {
    Writer(ResponseWriter),                                // for regular
    Handle(Rc<ResponseWriter>, Box<dyn AsyncKVMapHandle>), // for async
}

struct Connection {
    reader: RequestReader,
    writer_or_handle: WriterOrHandle,
}

impl Connection {
    fn writer(&mut self) -> &mut ResponseWriter {
        let WriterOrHandle::Writer(writer) = &mut self.writer_or_handle else {
            unreachable!();
        };
        writer
    }

    fn handle(&mut self) -> (&Rc<ResponseWriter>, &mut Box<dyn AsyncKVMapHandle>) {
        let WriterOrHandle::Handle(writer, handle) = &mut self.writer_or_handle else {
            unreachable!();
        };
        (writer, handle)
    }
}

type StreamMap = HashMap<Token, Connection>;

impl AsyncResponder for ResponseWriter {
    fn callback(&self, response: Response) {
        assert!(write_response(
            &mut *self.0.borrow_mut(),
            response.id,
            response.data.as_deref()
        )
        .is_ok());
    }
}

fn new_listener(host: &str, port: &str, nonblocking: bool) -> TcpListener {
    let addr: String = "".to_string() + host + ":" + port;
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        panic!("Server fails to bind address {}: {}", &addr, e);
    });
    assert!(listener.set_nonblocking(nonblocking).is_ok());
    listener
}

fn recv_requests(reader: &mut RequestReader) -> Vec<Request> {
    assert!(reader.fill_buf().is_ok());
    let mut requests = Vec::new();

    // here we must drain the buffer until it is empty, because one event may consist of multiple
    // batches, usually when the client keeps sending.
    while !reader.0.buffer().is_empty() {
        requests.append(&mut read_requests(reader).unwrap());
    }

    requests
}

fn server_worker_regular_main(
    worker_id: usize,
    poll: &mut Poll,
    events: &mut Events,
    smap: &mut StreamMap,
    handle: &mut Box<dyn KVMapHandle>,
    thread: &impl Thread,
) {
    for (_, connection) in smap.iter_mut() {
        assert!(connection.writer().flush().is_ok());
    }
    assert!(poll.poll(events, POLLING_TIMEOUT).is_ok());
    for event in events.iter() {
        let token = event.token();
        assert_ne!(token, Token(0));
        if event.is_read_closed() || event.is_write_closed() {
            assert!(smap.remove(&token).is_some());
        } else if event.is_error() {
            panic!("Server worker {} receives error event", worker_id);
        } else {
            if let Some(connection) = smap.get_mut(&token) {
                let requests = recv_requests(&mut connection.reader);
                serve_requests_regular(handle, &requests, connection.writer());
            } else {
                panic!("Server worker {} receives non-exist event", worker_id);
            }
        }
        thread.yield_now();
    }
}

fn server_worker_async_main(
    worker_id: usize,
    poll: &mut Poll,
    events: &mut Events,
    smap: &mut StreamMap,
    thread: &impl Thread,
) {
    for (_, connection) in smap.iter_mut() {
        let (writer, handle) = connection.handle();
        handle.drain();
        assert!(writer.0.borrow_mut().flush().is_ok());
    }
    assert!(poll.poll(events, POLLING_TIMEOUT).is_ok());
    for event in events as &Events {
        let token = event.token();
        assert_ne!(token, Token(0));
        if event.is_read_closed() || event.is_write_closed() {
            assert!(smap.remove(&token).is_some());
        } else if event.is_error() {
            panic!("Server worker {} receives error event", worker_id);
        } else {
            if let Some(connection) = smap.get_mut(&token) {
                let requests = recv_requests(&mut connection.reader);
                let handle = connection.handle().1;
                serve_requests_async(handle, &requests);
            } else {
                panic!("Server worker {} receives non-exist event", worker_id);
            }
        }
        thread.yield_now();
    }
}

fn server_worker_check_msg(
    listener: &Arc<TcpListener>,
    rx: &Receiver<WorkerMsg>,
    txs: &Vec<Sender<WorkerMsg>>,
    counter: &Arc<AtomicUsize>,
    nr_workers: usize,
) -> Option<WorkerMsg> {
    if let Ok((s, addr)) = listener.accept() {
        let w = counter.fetch_add(1, Ordering::AcqRel) % nr_workers;
        debug!("New connection dispatched to worker {}", w);
        assert!(txs[w].send(WorkerMsg::NewConnection(s, addr)).is_ok());
    }
    if let Ok(msg) = rx.try_recv() {
        return Some(msg);
    }
    None
}

fn server_worker_new_connection(
    stream: StdTcpStream,
    addr: SocketAddr,
    poll: &Poll,
) -> (Token, RequestReader, ResponseWriter) {
    let mut stream = TcpStream::from_std(stream);
    let token = Token(addr.port().into());
    assert!(poll
        .registry()
        .register(&mut stream, token, Interest::READABLE)
        .is_ok());
    let stream = RcTcpStream::new(stream);
    let reader = RequestReader::new(stream.clone());
    let writer = ResponseWriter::new(stream.clone());
    (token, reader, writer)
}

fn server_worker_common() -> (Events, StreamMap, Poll) {
    let events = Events::with_capacity(1024);
    let smap = StreamMap::new();
    let poll = Poll::new().unwrap();
    (events, smap, poll)
}

fn server_worker_regular(
    map: Arc<Box<impl KVMap + ?Sized>>,
    worker_id: usize,
    listener: Arc<TcpListener>,
    rx: Receiver<WorkerMsg>,
    txs: Vec<Sender<WorkerMsg>>,
    nr_workers: usize,
    counter: Arc<AtomicUsize>,
    thread: impl Thread,
) {
    thread.pin(worker_id);

    let (mut events, mut smap, mut poll) = server_worker_common();
    debug!("Server worker {} is ready", worker_id);

    let mut handle = map.handle();

    loop {
        if let Some(msg) = server_worker_check_msg(&listener, &rx, &txs, &counter, nr_workers) {
            match msg {
                WorkerMsg::Terminate => {
                    debug!("Server worker {} terminates", worker_id);
                    break;
                }
                WorkerMsg::NewConnection(s, addr) => {
                    let (token, reader, writer) = server_worker_new_connection(s, addr, &poll);
                    smap.insert(
                        token,
                        Connection {
                            reader,
                            writer_or_handle: WriterOrHandle::Writer(writer),
                        },
                    );
                }
            }
        }
        server_worker_regular_main(
            worker_id,
            &mut poll,
            &mut events,
            &mut smap,
            &mut handle,
            &thread,
        );
        thread.yield_now();
    }
}

fn server_worker_async(
    map: Arc<Box<impl AsyncKVMap + ?Sized>>,
    worker_id: usize,
    listener: Arc<TcpListener>,
    rx: Receiver<WorkerMsg>,
    txs: Vec<Sender<WorkerMsg>>,
    nr_workers: usize,
    counter: Arc<AtomicUsize>,
    thread: impl Thread,
) {
    thread.pin(worker_id);

    let (mut events, mut smap, mut poll) = server_worker_common();
    debug!("Server worker {} is ready", worker_id);

    loop {
        if let Some(msg) = server_worker_check_msg(&listener, &rx, &txs, &counter, nr_workers) {
            match msg {
                WorkerMsg::Terminate => {
                    debug!("Server worker {} terminates", worker_id);
                    break;
                }
                WorkerMsg::NewConnection(s, addr) => {
                    let (token, reader, writer) = server_worker_new_connection(s, addr, &poll);
                    let writer = Rc::new(writer);
                    let handle = map.handle(writer.clone());
                    smap.insert(
                        token,
                        Connection {
                            reader,
                            writer_or_handle: WriterOrHandle::Handle(writer, handle),
                        },
                    );
                }
            }
        }
        server_worker_async_main(worker_id, &mut poll, &mut events, &mut smap, &thread);
        thread.yield_now();
    }
}

fn server_common(
    host: &str,
    port: &str,
    nr_workers: usize,
) -> (
    Arc<TcpListener>,
    Vec<Sender<WorkerMsg>>,
    Vec<Receiver<WorkerMsg>>,
    Arc<AtomicUsize>,
) {
    let listener = Arc::new(new_listener(host, port, true));

    let mut senders = Vec::<Sender<WorkerMsg>>::with_capacity(nr_workers);
    let mut receivers = Vec::<Receiver<WorkerMsg>>::with_capacity(nr_workers);

    // create channels
    for _ in 0..nr_workers {
        let (tx, rx) = channel();
        senders.push(tx);
        receivers.push(rx);
    }
    let counter = Arc::new(AtomicUsize::new(0));

    return (listener, senders, receivers, counter);
}

fn server_mainloop(
    stop_rx: Receiver<()>,
    grace_tx: Sender<()>,
    senders: Vec<Sender<WorkerMsg>>,
    mut handles: Vec<impl JoinHandle>,
    thread: impl Thread,
) {
    loop {
        if let Ok(_) = stop_rx.try_recv() {
            break;
        }
        thread.yield_now();
    }
    let nr_workers = handles.len();
    for i in 0..nr_workers {
        assert!(senders[i].send(WorkerMsg::Terminate).is_ok());
    }
    while let Some(handle) = handles.pop() {
        handle.join();
    }
    assert!(grace_tx.send(()).is_ok());
}

/// The real server function for [`KVMap`].
///
/// **You may not need to check this if it is OK to run benchmarks with [`std::thread`].**
pub fn server_regular(
    map: Arc<Box<impl KVMap + ?Sized>>,
    host: &str,
    port: &str,
    nr_workers: usize,
    stop_rx: Receiver<()>,
    grace_tx: Sender<()>,
    thread: impl Thread,
) {
    let (listener, senders, mut receivers, counter) = server_common(host, port, nr_workers);

    let mut handles = Vec::new();
    for i in 0..nr_workers {
        let map = map.clone();
        let listener = listener.clone();
        let txs: Vec<Sender<WorkerMsg>> = (0..nr_workers).map(|w| senders[w].clone()).collect();
        let rx = receivers.pop().unwrap(); // guaranteed to succeed
        let nr_workers = nr_workers.clone();
        let counter = counter.clone();
        let worker_thread = thread.clone();
        let handle = thread.spawn(move || {
            server_worker_regular(
                map,
                i,
                listener,
                rx,
                txs,
                nr_workers,
                counter,
                worker_thread,
            );
        });
        handles.push(handle);
    }

    server_mainloop(stop_rx, grace_tx, senders, handles, thread);
}

/// The real server function for [`AsyncKVMap`].
///
/// **You may not need to check this if it is OK to run benchmarks with [`std::thread`].**
pub fn server_async(
    map: Arc<Box<impl AsyncKVMap + ?Sized>>,
    host: &str,
    port: &str,
    nr_workers: usize,
    stop_rx: Receiver<()>,
    grace_tx: Sender<()>,
    thread: impl Thread,
) {
    let (listener, senders, mut receivers, counter) = server_common(host, port, nr_workers);

    let mut handles = Vec::new();
    for i in 0..nr_workers {
        let map = map.clone();
        let listener = listener.clone();
        let txs: Vec<Sender<WorkerMsg>> = (0..nr_workers).map(|w| senders[w].clone()).collect();
        let rx = receivers.pop().unwrap(); // guaranteed to succeed
        let nr_workers = nr_workers.clone();
        let counter = counter.clone();
        let worker_thread = thread.clone();
        let handle = thread.spawn(move || {
            server_worker_async(
                map,
                i,
                listener,
                rx,
                txs,
                nr_workers,
                counter,
                worker_thread,
            );
        });
        handles.push(handle);
    }

    server_mainloop(stop_rx, grace_tx, senders, handles, thread);
}

pub(crate) struct KVClient {
    request_writer: BufWriter<TcpStream>,
    response_reader: BufReader<TcpStream>,
}

impl KVClient {
    pub(crate) fn new(host: &str, port: &str) -> Option<Self> {
        let addr: String = "".to_string() + host + ":" + port;
        match StdTcpStream::connect(&addr) {
            Ok(s) => {
                let s2 = s.try_clone().unwrap_or_else(|e| {
                    panic!("KVClient fails to clone a tcp stream: {}", e);
                });
                Some(KVClient {
                    request_writer: BufWriter::new(TcpStream::from_std(s)),
                    response_reader: BufReader::new(TcpStream::from_std(s2)),
                })
            }
            Err(_) => None,
        }
    }

    pub(crate) fn send_requests(&mut self, requests: &Vec<Request>) {
        assert!(write_requests(&mut self.request_writer, requests).is_ok());
        assert!(self.request_writer.flush().is_ok());
    }

    // recv all (drain the buffer)
    pub(crate) fn recv_responses(&mut self) -> Vec<Response> {
        assert!(self.response_reader.fill_buf().is_ok());
        let mut responses = Vec::new();

        while !self.response_reader.buffer().is_empty() {
            match read_response(&mut self.response_reader) {
                Ok(r) => {
                    responses.push(r);
                }
                Err(e) => {
                    panic!("KVClient failed to read response: {}", e);
                }
            }
        }

        responses
    }

    #[cfg(test)]
    fn recv_responses_n(&mut self, nr: usize) -> Vec<Response> {
        let mut responses = Vec::<Response>::with_capacity(nr);

        for _ in 0..nr {
            match read_response(&mut self.response_reader) {
                Ok(r) => {
                    responses.push(r);
                }
                Err(e) => {
                    panic!("KVClient failed to read response: {}", e);
                }
            }
        }

        responses
    }

    #[cfg(test)]
    fn set(&mut self, key: &[u8], value: &[u8]) {
        let mut requests = Vec::<Request>::with_capacity(1);
        let op = Operation::Set {
            key: key.into(),
            value: value.into(),
        };
        requests.push(Request { id: 0, op });
        self.send_requests(&requests);

        let mut responses = self.recv_responses_n(1);
        assert_eq!(responses.len(), 1);
        let response = responses.pop().unwrap();
        assert_eq!(response.id, 0);
        assert!(response.data.is_none());
    }

    #[cfg(test)]
    fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        let mut requests = Vec::<Request>::with_capacity(1);
        let op = Operation::Get { key: key.into() };
        requests.push(Request { id: 0, op });
        self.send_requests(&requests);

        let mut responses = self.recv_responses_n(1);
        assert_eq!(responses.len(), 1);
        let response = responses.pop().unwrap();
        assert_eq!(response.id, 0);
        match response.data {
            Some(v) => Some(v),
            None => None,
        }
    }

    #[cfg(test)]
    fn delete(&mut self, key: &[u8]) {
        let mut requests = Vec::<Request>::with_capacity(1);
        let op = Operation::Delete { key: key.into() };
        requests.push(Request { id: 0, op });
        self.send_requests(&requests);

        let mut responses = self.recv_responses_n(1);
        assert_eq!(responses.len(), 1);
        let response = responses.pop().unwrap();
        assert_eq!(response.id, 0);
        assert!(response.data.is_none());
    }
}

#[derive(Deserialize, Debug)]
struct ServerMapOpt {
    map: BenchKVMapOpt,
}

pub fn init(text: &str) -> BenchKVMap {
    let opt: ServerMapOpt = Figment::new()
        .merge(Toml::string(&text))
        .merge(Env::raw())
        .extract()
        .unwrap();
    BenchKVMap::new(&opt.map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stores::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::time::Duration;

    static PORT: AtomicU32 = AtomicU32::new(9000);

    fn addr() -> (String, String, String) {
        let host = "127.0.0.1".to_string();
        let port = PORT.fetch_add(1, Ordering::AcqRel).to_string();
        let addr = "".to_owned() + &host + ":" + &port;
        (host, port, addr)
    }

    fn server_run(
        map: BenchKVMap,
        host: &str,
        port: &str,
        nr_workers: usize,
    ) -> (Sender<()>, Receiver<()>) {
        let _ = env_logger::try_init();
        let (host, port) = (host.to_string(), port.to_string());
        let (stop_tx, stop_rx) = channel();
        let (grace_tx, grace_rx) = channel();
        let _ = std::thread::spawn(move || match map {
            BenchKVMap::Regular(map) => {
                map.server(&host, &port, nr_workers, stop_rx, grace_tx);
            }
            BenchKVMap::Async(map) => {
                map.server(&host, &port, nr_workers, stop_rx, grace_tx);
            }
        });
        std::thread::sleep(Duration::from_millis(1000));
        (stop_tx, grace_rx)
    }

    fn simple(map: BenchKVMap) {
        let (host, port, _) = addr();
        let (stop_tx, grace_rx) = server_run(map, &host, &port, 4);
        let mut client = KVClient::new(&host, &port)
            .unwrap_or_else(|| panic!("failed to unwrap client instance"));

        client.set(b"foo", b"bar");
        assert_eq!(client.get(b"foo").unwrap(), (*b"bar").into());
        assert!(client.get(b"f00").is_none());

        client.set(b"foo", b"car");
        assert_eq!(client.get(b"foo").unwrap(), (*b"car").into());

        client.delete(b"foo");
        assert!(client.get(b"foo").is_none());

        assert!(stop_tx.send(()).is_ok());
        assert!(grace_rx.recv().is_ok());
    }

    #[test]
    fn simple_mutex() {
        let opt = hashmap::MutexHashMapOpt { shards: 512 };
        let map = BenchKVMap::Regular(Box::new(hashmap::MutexHashMap::new(&opt)));
        simple(map);
    }

    #[test]
    fn simple_rwlock() {
        let opt = hashmap::RwLockHashMapOpt { shards: 512 };
        let map = BenchKVMap::Regular(Box::new(hashmap::RwLockHashMap::new(&opt)));
        simple(map);
    }

    #[test]
    fn simple_dashmap() {
        let map = BenchKVMap::Regular(Box::new(dashmap::DashMap::new()));
        simple(map);
    }

    #[test]
    fn simple_contrie() {
        let map = BenchKVMap::Regular(Box::new(contrie::Contrie::new()));
        simple(map);
    }

    #[test]
    fn simple_chashmap() {
        let map = BenchKVMap::Regular(Box::new(chashmap::CHashMap::new()));
        simple(map);
    }

    #[test]
    fn simple_scchashmap() {
        let map = BenchKVMap::Regular(Box::new(scc::SccHashMap::new()));
        simple(map);
    }

    #[test]
    fn simple_flurry() {
        let map = BenchKVMap::Regular(Box::new(flurry::Flurry::new()));
        simple(map);
    }

    #[test]
    fn simple_papaya() {
        let map = BenchKVMap::Regular(Box::new(papaya::Papaya::new()));
        simple(map);
    }

    fn batch(map: BenchKVMap) {
        const NR_CLIENTS: usize = 8;
        const NR_BATCHES: usize = 1000;
        const BATCH_SIZE: usize = 100;

        assert_eq!(BATCH_SIZE % 2, 0);

        let mut requests = Vec::<Vec<Vec<Request>>>::with_capacity(NR_CLIENTS);
        for i in 0..NR_CLIENTS {
            let mut seq = 0;
            requests.push(Vec::<Vec<Request>>::with_capacity(NR_BATCHES));
            for j in 0..NR_BATCHES {
                requests[i].push(Vec::<Request>::with_capacity(BATCH_SIZE));
                for k in 0..BATCH_SIZE / 2 {
                    let op1 = Operation::Set {
                        key: format!("{}", k + j * BATCH_SIZE + i * NR_BATCHES * BATCH_SIZE)
                            .as_bytes()
                            .into(),
                        value: [170u8; 16].into(),
                    };
                    let op2 = Operation::Get {
                        key: format!("{}", k + j * BATCH_SIZE + i * NR_BATCHES * BATCH_SIZE)
                            .as_bytes()
                            .into(),
                    };
                    requests[i][j].push(Request { id: seq, op: op1 });
                    requests[i][j].push(Request {
                        id: seq + 1,
                        op: op2,
                    });
                    seq += 2;
                }
            }
        }

        let (host, port, _) = addr();
        let (stop_tx, grace_rx) = server_run(map, &host, &port, 8);

        for i in 0..NR_CLIENTS {
            let mut client = KVClient::new(&host, &port)
                .unwrap_or_else(|| panic!("failed to create client instance"));
            let batch = requests[i].clone();
            let mut pending: usize = 0;
            for j in 0..NR_BATCHES {
                client.send_requests(&batch[j]);
                pending += BATCH_SIZE;
                loop {
                    let response = client.recv_responses();
                    for r in response {
                        let id = r.id;
                        if id % 2 == 0 {
                            // set
                            assert_eq!(r.data, None);
                        } else {
                            assert_eq!(r.data, Some([170u8; 16].into()));
                        }
                        pending -= 1;
                    }
                    if pending < BATCH_SIZE * 10 {
                        break;
                    }
                }
            }
            // finish remaining
            loop {
                if pending == 0 {
                    break;
                }
                let response = client.recv_responses_n(pending);
                for r in response {
                    let id = r.id;
                    if id % 2 == 0 {
                        // set
                        assert_eq!(r.data, None);
                    } else {
                        assert_eq!(r.data, Some([170u8; 16].into()));
                    }
                    pending -= 1;
                }
            }
        }

        assert!(stop_tx.send(()).is_ok());
        assert!(grace_rx.recv().is_ok());
    }

    #[test]
    fn batch_mutex_hashmap() {
        let opt = hashmap::MutexHashMapOpt { shards: 512 };
        let map = BenchKVMap::Regular(Box::new(hashmap::MutexHashMap::new(&opt)));
        batch(map);
    }

    #[test]
    fn batch_rwlock_hashmap() {
        let opt = hashmap::RwLockHashMapOpt { shards: 512 };
        let map = BenchKVMap::Regular(Box::new(hashmap::RwLockHashMap::new(&opt)));
        batch(map);
    }

    #[test]
    fn batch_dashmap() {
        let map = BenchKVMap::Regular(Box::new(dashmap::DashMap::new()));
        batch(map);
    }

    #[test]
    fn batch_contrie() {
        let map = BenchKVMap::Regular(Box::new(contrie::Contrie::new()));
        batch(map);
    }

    #[test]
    fn batch_chashmap() {
        let map = BenchKVMap::Regular(Box::new(chashmap::CHashMap::new()));
        batch(map);
    }

    #[test]
    fn batch_scchashmap() {
        let map = BenchKVMap::Regular(Box::new(scc::SccHashMap::new()));
        batch(map);
    }

    #[test]
    fn batch_flurry() {
        let map = BenchKVMap::Regular(Box::new(flurry::Flurry::new()));
        batch(map);
    }

    #[test]
    fn batch_papaya() {
        let map = BenchKVMap::Regular(Box::new(papaya::Papaya::new()));
        batch(map);
    }

    #[test]
    fn batch_mutex_btreemap() {
        let map = BenchKVMap::Regular(Box::new(btreemap::MutexBTreeMap::new()));
        batch(map);
    }

    #[test]
    fn batch_rwlock_btreemap() {
        let map = BenchKVMap::Regular(Box::new(btreemap::RwLockBTreeMap::new()));
        batch(map);
    }
}

use crate::bench::*;
use crate::serialization::{read_request, write_response};
use crate::thread::{JoinHandle, Thread};
use crate::*;
use crate::{AsyncKVMap, AsyncKVMapHandle, KVMap, KVMapHandle, Operation, Request};
use clap::Parser;
use hashbrown::HashMap;
use log::debug;
use mio::net::TcpStream;
use mio::{Events, Interest, Poll, Token};
use std::cell::RefCell;
use std::fs::read_to_string;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream as StdTcpStream};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

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
        }
    }
}

pub(crate) fn serve_requests_async(
    handle: &mut Box<dyn AsyncKVMapHandle>,
    requests: &Vec<Request>,
) {
    handle.submit(requests);
}

/// Wrapper around TcpStream to enable multi-ownership in reader/writer for the same connection
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
        Self(BufReader::with_capacity(8usize << 20, stream))
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
/// and potentially a wrapper rc.
struct ResponseWriter(RefCell<BufWriter<RcTcpStream>>);

impl ResponseWriter {
    fn new(stream: RcTcpStream) -> Self {
        Self(RefCell::new(BufWriter::with_capacity(8usize << 20, stream)))
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

    while !reader.0.buffer().is_empty() {
        let reader = &mut *reader;
        let request = read_request(reader).unwrap();
        requests.push(request);
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
    let (mut events, mut smap, mut poll) = server_worker_common();
    debug!("Server worker {} is ready", worker_id);

    let mut handle = map.handle();

    loop {
        if let Some(msg) = server_worker_check_msg(&listener, &rx, &txs, &counter, nr_workers) {
            match msg {
                WorkerMsg::Terminate => break,
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
    let (mut events, mut smap, mut poll) = server_worker_common();
    debug!("Server worker {} is ready", worker_id);

    loop {
        if let Some(msg) = server_worker_check_msg(&listener, &rx, &txs, &counter, nr_workers) {
            match msg {
                WorkerMsg::Terminate => break,
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

#[derive(Deserialize, Debug)]
struct ServerMapOpt {
    map: BenchKVMapOpt,
}

pub fn cli() {
    env_logger::init();

    #[derive(Parser, Debug)]
    #[command(about)]
    struct Args {
        #[arg(long, short = 'h', default_value = "0.0.0.0")]
        host: String,

        #[arg(long, short = 'p', default_value = "9000")]
        port: String,

        #[arg(long, short = 'm')]
        map_file: Option<String>,

        #[arg(long, short = 'n')]
        workers: usize,
    }

    let args = Args::parse();
    debug!("Starting server with args: {:?}", args);

    let host = &args.host;
    let port = &args.port;
    let nr_workers = args.workers;

    let opt: String = read_to_string(args.map_file.as_ref().unwrap().as_str()).unwrap();
    let opt: ServerMapOpt = toml::from_str(&opt).unwrap();
    let map = BenchKVMap::new(&opt.map);

    let (_stop_tx, stop_rx) = channel();
    let (grace_tx, _grace_rx) = channel();

    match map {
        BenchKVMap::Regular(map) => {
            map.server(&host, &port, nr_workers, stop_rx, grace_tx);
        }
        BenchKVMap::Async(map) => {
            map.server(&host, &port, nr_workers, stop_rx, grace_tx);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::bench::BenchKVMap;
    use crate::client::KVClient;
    use crate::map::*;
    use crate::*;
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

        assert!(client.set(b"foo", b"bar").is_ok());
        assert_eq!(client.get(b"foo").unwrap(), (*b"bar").into());
        assert!(client.get(b"f00").is_err());
        assert!(client.set(b"foo", b"car").is_ok());
        assert_eq!(client.get(b"foo").unwrap(), (*b"car").into());

        assert!(stop_tx.send(()).is_ok());
        assert!(grace_rx.recv().is_ok());
    }

    #[test]
    fn simple_mutex() {
        let opt = MutexHashMapOpt { shards: 512 };
        let map = BenchKVMap::Regular(Box::new(MutexHashMap::new(&opt)));
        simple(map);
    }

    #[test]
    fn simple_rwlock() {
        let opt = RwLockHashMapOpt { shards: 512 };
        let map = BenchKVMap::Regular(Box::new(RwLockHashMap::new(&opt)));
        simple(map);
    }

    #[test]
    fn simple_dashmap() {
        let map = BenchKVMap::Regular(Box::new(DashMap::new()));
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
                // println!("send: client {} batch {} pending {}", i, j, pending);
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
                // println!("recv: client {} batch {} pending {}", i, j, pending);
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
    fn batch_mutex() {
        let opt = MutexHashMapOpt { shards: 512 };
        let map = BenchKVMap::Regular(Box::new(MutexHashMap::new(&opt)));
        batch(map);
    }

    #[test]
    fn batch_rwlock() {
        let opt = RwLockHashMapOpt { shards: 512 };
        let map = BenchKVMap::Regular(Box::new(RwLockHashMap::new(&opt)));
        batch(map);
    }

    #[test]
    fn batch_dashmap() {
        let map = BenchKVMap::Regular(Box::new(DashMap::new()));
        batch(map);
    }
}

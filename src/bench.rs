use crate::thread::{JoinHandle, Thread};
use crate::workload::{Workload, WorkloadOpt};
use crate::*;
use clap::Parser;
use hashbrown::HashMap;
use log::debug;
use quanta::Instant;
use serde::Deserialize;
use std::fs::read_to_string;
use std::rc::Rc;
use std::sync::Mutex;
use std::sync::{Arc, Barrier};
use std::time::Duration;

// {{{ benchmap

/// The bencher supports two types of maps: regular (sync) and async,
/// and they correspond to KVMap and AsyncKVMap backed store, respectively.
pub enum BenchKVMap {
    Regular(Box<dyn KVMap>),
    Async(Box<dyn AsyncKVMap>),
}

impl BenchKVMap {
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

/// We would like to make the benchmark module as extendable as possible. That is, when adding a
/// new beckend map/collection, we actually do not want to modify anything in the benchmark
/// framework. The user then can directly use this crate as a dependency then do whatever they want
/// to do. So we just use a generic toml map structure to encode all map-specific configurations.
/// When a new kvmap is added, we use the inventory crate to dynamically register them.
pub struct Registry<'a> {
    name: &'a str,
    constructor: fn(&toml::Table) -> BenchKVMap,
}

impl<'a> Registry<'a> {
    pub const fn new(name: &'a str, constructor: fn(&toml::Table) -> BenchKVMap) -> Self {
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
    opt: toml::Table,
}

impl BenchKVMap {
    pub(crate) fn new(opt: &BenchKVMapOpt) -> BenchKVMap {
        // construct the hashmap.. this will be done every time
        let mut registered: HashMap<&'static str, fn(&toml::Table) -> BenchKVMap> = HashMap::new();
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

// }}} benchmap

// {{{ benchmark

/// Length determines when a benchmark should stop or how often the metrics should be collected.
#[derive(Clone, Debug)]
enum Length {
    /// Each worker thread syncs after a timeout (e.g., 0.1s).
    Timeout(Duration),
    /// Each worker thread syncs after a number of operations (e.g., 1M operations ea.).
    Count(u64),
    /// Special: exhaust the number of keys in the key space (max - min)
    Exhaust,
}

/// How the results are printed out.
/// "hidden": no results
/// "repeat": only each repeat's own metrics
/// "finish": only the finish metrics
/// "all": equals to repeat + finish
#[derive(Debug, PartialEq)]
enum ReportMode {
    Hidden,
    Repeat,
    Finish,
    All,
}

/// The configuration of a single benchmark deserialized from a toml string. The fields are
/// optional to ease parsing from toml, as there can be global parameters that are set for them.
#[derive(Deserialize, Clone, Debug)]
struct BenchmarkOpt {
    /// Number of threads that runs this benchmark.
    threads: Option<usize>,
    /// How many times this benchmark will be executed.
    repeat: Option<usize>,
    /// How long this benchmark will run, unit is seconds.
    timeout: Option<f32>,
    /// Fallback bound when timeout is not given.
    ops: Option<u64>,
    /// Report mode: "hidden", "phase", "finish", "all"
    report: Option<String>,
    /// Max depth of queue for each worker (async only)
    qd: Option<usize>,
    /// Batch size for each request (async only)
    batch: Option<usize>,
    /// The definition of a workload. (flattened)
    #[serde(flatten)]
    workload: WorkloadOpt,
}

/// Instantiated from BenchmarkOpt
#[derive(Debug)]
pub struct Benchmark {
    threads: usize,
    repeat: usize,
    qd: usize,
    batch: usize,
    len: Length,
    report: ReportMode,
    wopt: WorkloadOpt,
}

const TIME_CHECK_INTERVAL: u64 = 128;

impl Benchmark {
    /// The constructor of Benchmark expects all fields have their values, the struct should
    /// contain either its own parameters, or carry the default parameters.
    fn new(opt: &BenchmarkOpt) -> Self {
        let threads = opt.threads.unwrap();
        let repeat = opt.repeat.unwrap();
        let qd = opt.qd.unwrap();
        let batch = opt.batch.unwrap();
        // handle length in the following, now 3 modes
        let len = if let Some(t) = opt.timeout {
            assert!(opt.ops.is_none());
            Length::Timeout(Duration::from_secs_f32(t))
        } else if let Some(c) = opt.ops {
            Length::Count(c)
        } else {
            Length::Exhaust
        };
        let report = match opt.report.as_ref().unwrap().as_str() {
            "hidden" => ReportMode::Hidden,
            "repeat" => ReportMode::Repeat,
            "finish" => ReportMode::Finish,
            "all" => ReportMode::All,
            _ => unreachable!(),
        };
        let wopt = opt.workload.clone();
        Self {
            threads,
            repeat,
            qd,
            batch,
            len,
            report,
            wopt,
        }
    }
}

// }}} benchmark

// {{{ benchmarkgroup

/// The global options that go to the [global] section in a BenchmarkGroup.
/// They will override missing fields.
#[derive(Deserialize, Clone, Debug)]
struct GlobalOpt {
    /// For benchmark
    threads: Option<usize>,
    repeat: Option<usize>,
    qd: Option<usize>,
    batch: Option<usize>,
    report: Option<String>,

    /// For workloads
    klen: Option<usize>,
    vlen: Option<usize>,
    kmin: Option<usize>,
    kmax: Option<usize>,
}

impl Default for GlobalOpt {
    fn default() -> Self {
        Self {
            threads: None,
            repeat: None,
            qd: None,
            batch: None,
            report: None,
            klen: None,
            vlen: None,
            kmin: None,
            kmax: None,
        }
    }
}

impl GlobalOpt {
    fn apply(&self, opt: &mut BenchmarkOpt) {
        // benchmark itself (these fall back to defaults)
        opt.threads = opt.threads.or_else(|| Some(self.threads.unwrap_or(1)));
        opt.repeat = opt.repeat.or_else(|| Some(self.repeat.unwrap_or(1)));
        opt.qd = opt.qd.or_else(|| Some(self.qd.unwrap_or(1)));
        opt.batch = opt.batch.or_else(|| Some(self.batch.unwrap_or(1)));
        opt.report = opt
            .report
            .clone()
            .or_else(|| Some(self.report.clone().unwrap_or("all".to_string())));
        // the workload options (must be specified)
        opt.workload.klen = opt
            .workload
            .klen
            .or_else(|| Some(self.klen.expect("klen should be given")));
        opt.workload.vlen = opt
            .workload
            .vlen
            .or_else(|| Some(self.vlen.expect("vlen should be given")));
        opt.workload.kmin = opt
            .workload
            .kmin
            .or_else(|| Some(self.kmin.expect("kmin should be given")));
        opt.workload.kmax = opt
            .workload
            .kmax
            .or_else(|| Some(self.kmax.expect("kmax should be given")));
    }
}

/// The configuration of a group of benchmark(s). It has a global option that could possibly
/// override benchmark-local options.
#[derive(Deserialize, Clone, Debug)]
struct BenchmarkGroupOpt {
    /// Global parameters (optional)
    global: Option<GlobalOpt>,

    /// Map configuration
    map: BenchKVMapOpt,

    /// Array of the parameters of consisting Benchmark(s)
    benchmark: Vec<BenchmarkOpt>,
}

// }}} benchmarkgroup

// {{{ bencher

pub fn init(text: &str) -> (BenchKVMap, Vec<Arc<Benchmark>>) {
    let opt: BenchmarkGroupOpt = toml::from_str(text).unwrap();
    debug!(
        "Creating benchmark group with the following configurations: {:?}",
        opt
    );
    let global = opt.global.clone().unwrap_or_default();
    // now we have a bunch of BenchmarkOpt(s), we need to update their params if they did
    // not specify, using the default values given. If all are missing, it's a panic.
    let mut bopts: Vec<BenchmarkOpt> = opt.benchmark.iter().map(|o| o.clone()).collect();
    for bopt in bopts.iter_mut() {
        global.apply(bopt);
    }
    debug!("Global options applied to benchmarks: {:?}", bopts);
    // this instance of map is actually not used - the sole purpose is to get a handle out of
    // it later in each phase
    let map = BenchKVMap::new(&opt.map);
    let phases = bopts
        .into_iter()
        .map(|o| Arc::new(Benchmark::new(&o)))
        .collect();
    (map, phases)
}

#[inline]
fn bench_phase_should_break(
    len: &Length,
    count: &u64,
    start: &Instant,
    workload: &Workload,
) -> bool {
    match len {
        Length::Count(c) => {
            if count == c {
                return true;
            }
        }
        Length::Timeout(duration) => {
            // only checks after a certain interval
            if count % TIME_CHECK_INTERVAL == 0 {
                if Instant::now().duration_since(*start) >= *duration {
                    return true;
                }
            }
        }
        Length::Exhaust => {
            if workload.is_exhausted() {
                return true;
            }
        }
    }
    false
}

fn bench_worker_regular(
    map: Arc<Box<impl KVMap + ?Sized>>,
    benchmark: Arc<Benchmark>,
    barrier: Arc<Barrier>,
    counter: Arc<Mutex<u64>>,
    thread_info: (usize, usize),
    thread: impl Thread,
) {
    thread.pin(thread_info.0);

    let mut handle = map.handle();
    let mut rng = rand::thread_rng();
    let mut workload = Workload::new(&benchmark.wopt, Some(thread_info));
    for _ in 0..benchmark.repeat {
        let mut count = 0u64;
        // first sync, wait for main thread complete collecting data
        barrier.wait();
        *counter.lock().unwrap() = 0u64; // reset
        let start = Instant::now();
        // start benchmark
        loop {
            let op = workload.next(&mut rng);
            match op {
                Operation::Set { key, value } => {
                    handle.set(&key[..], &value[..]);
                }
                Operation::Get { key } => {
                    let _ = handle.get(&key[..]);
                }
            }
            count += 1;
            // check if we need to break
            if bench_phase_should_break(&benchmark.len, &count, &start, &workload) {
                break;
            }
        }
        // after the loop, update counter
        *counter.lock().unwrap() = count;
        // barrier 2
        barrier.wait();
    }
}

fn bench_worker_async(
    map: Arc<Box<impl AsyncKVMap + ?Sized>>,
    benchmark: Arc<Benchmark>,
    barrier: Arc<Barrier>,
    counter: Arc<Mutex<u64>>,
    thread_info: (usize, usize),
    thread: impl Thread,
) {
    thread.pin(thread_info.0);

    let responder = Rc::new(RefCell::new(Vec::<Response>::new()));
    let mut handle = map.handle(responder.clone());
    let mut rng = rand::thread_rng();
    let mut workload = Workload::new(&benchmark.wopt, Some(thread_info));
    // pending requests is global, as it is not needed to drain all requests after each repeat
    let mut pending = 0usize;
    let mut requests = Vec::<Request>::with_capacity(benchmark.batch);
    let mut id = 0usize;
    for _ in 0..benchmark.repeat {
        let mut count = 0u64;
        // first sync, wait for main thread complete collecting data
        barrier.wait();
        *counter.lock().unwrap() = 0u64; // reset
        let start = Instant::now();
        // start benchmark
        loop {
            // first clear the requests vector
            requests.clear();
            // sample requests
            for _ in 0..benchmark.batch {
                let op = workload.next(&mut rng);
                requests.push(Request { id, op });
                id += 1;
                // need to add to count here, instead of after this loop
                // otherwise the last check may fail because the time check is after a certain
                // interval, but the mod is never 0
                count += 1;
                if bench_phase_should_break(&benchmark.len, &count, &start, &workload) {
                    break;
                }
            }
            // now we have a batch, send it all, whatever its size is
            let len = requests.len();
            handle.submit(&requests);
            pending += len;
            // use a loop to make sure that pending is under qd
            loop {
                handle.drain();
                let responses = responder.replace_with(|_| Vec::new());
                pending -= responses.len();
                if pending <= benchmark.qd {
                    break;
                }
            }
            if bench_phase_should_break(&benchmark.len, &count, &start, &workload) {
                break;
            }
        }
        // after the loop, update counter
        *counter.lock().unwrap() = count;
        // barrier 2
        barrier.wait();
    }
    loop {
        if pending == 0 {
            break;
        }
        handle.drain();
        let responses = responder.replace_with(|_| Vec::new());
        pending -= responses.len();
    }
}

fn bench_mainloop(
    benchmark: Arc<Benchmark>,
    barrier: Arc<Barrier>,
    phase: usize,
    since: Instant,
    seq: &mut usize,
    counters: Vec<Arc<Mutex<u64>>>,
    mut handles: Vec<impl JoinHandle>,
) {
    let mut total_duration = 0f64;
    let mut total_ops = 0u64;
    for i in 0..benchmark.repeat {
        barrier.wait();
        let start = Instant::now();
        // worker runs
        barrier.wait();
        let end = Instant::now();
        let duration = end - start;
        let elapsed = end - since;
        // collecting data
        let mut c = 0;
        for i in counters.iter() {
            c += *i.lock().unwrap();
        }
        if benchmark.report == ReportMode::Repeat || benchmark.report == ReportMode::All {
            println!(
                "{} phase {} repeat {} duration {:.2} elapsed {:.2} total {} mops {:.2}",
                *seq,
                phase,
                i,
                duration.as_secs_f64(),
                elapsed.as_secs_f64(),
                c,
                c as f64 / duration.as_secs_f64() / 1000000.0
            );
            *seq += 1;
        }
        total_duration += duration.as_secs_f64();
        total_ops += c;
    }

    while let Some(handle) = handles.pop() {
        handle.join();
    }

    if benchmark.report == ReportMode::Finish || benchmark.report == ReportMode::All {
        let total_elapsed = Instant::now() - since;
        println!(
            "{} phase {} finish . duration {:.2} elapsed {:.2} total {} mops {:.2}",
            *seq,
            phase,
            total_duration,
            total_elapsed.as_secs_f64(),
            total_ops,
            total_ops as f64 / total_duration / 1000000.0
        );
        *seq += 1;
    }
}

fn bench_phase_regular(
    map: Arc<Box<impl KVMap + ?Sized>>,
    benchmark: Arc<Benchmark>,
    phase: usize,
    since: Instant,
    seq: &mut usize,
    thread: &impl Thread,
) {
    let barrier = Arc::new(Barrier::new((benchmark.threads + 1).try_into().unwrap()));
    let counters: Vec<Arc<Mutex<u64>>> = (0..benchmark.threads)
        .map(|_| Arc::new(Mutex::new(0u64)))
        .collect();
    let mut handles = Vec::new();
    for t in 0..benchmark.threads {
        let map = map.clone();
        let benchmark = benchmark.clone();
        let barrier = barrier.clone();
        let counter = counters[t].clone();
        let thread_info = (t, benchmark.threads);
        let worker_thread = thread.clone();
        let handle = thread.spawn(move || {
            bench_worker_regular(map, benchmark, barrier, counter, thread_info, worker_thread);
        });
        handles.push(handle);
    }
    bench_mainloop(benchmark, barrier, phase, since, seq, counters, handles);
}

fn bench_phase_async(
    map: Arc<Box<impl AsyncKVMap + ?Sized>>,
    benchmark: Arc<Benchmark>,
    phase: usize,
    since: Instant,
    seq: &mut usize,
    thread: &impl Thread,
) {
    let barrier = Arc::new(Barrier::new((benchmark.threads + 1).try_into().unwrap()));
    let counters: Vec<Arc<Mutex<u64>>> = (0..benchmark.threads)
        .map(|_| Arc::new(Mutex::new(0u64)))
        .collect();
    let mut handles = Vec::new();
    for t in 0..benchmark.threads {
        let map = map.clone();
        let benchmark = benchmark.clone();
        let barrier = barrier.clone();
        let counter = counters[t].clone();
        let thread_info = (t, benchmark.threads);
        let worker_thread = thread.clone();
        let handle = thread.spawn(move || {
            bench_worker_async(map, benchmark, barrier, counter, thread_info, worker_thread);
        });
        handles.push(handle);
    }
    bench_mainloop(benchmark, barrier, phase, since, seq, counters, handles);
}

pub fn bench_regular(
    map: Arc<Box<impl KVMap + ?Sized>>,
    phases: &Vec<Arc<Benchmark>>,
    thread: impl Thread,
) {
    debug!("Running regular bencher");
    let start = Instant::now();
    let mut seq = 0usize;
    for (i, p) in phases.iter().enumerate() {
        bench_phase_regular(map.clone(), p.clone(), i, start.clone(), &mut seq, &thread);
    }
}

pub fn bench_async(
    map: Arc<Box<impl AsyncKVMap + ?Sized>>,
    phases: &Vec<Arc<Benchmark>>,
    thread: impl Thread,
) {
    debug!("Running async bencher");
    let start = Instant::now();
    let mut seq = 0usize;
    for (i, p) in phases.iter().enumerate() {
        bench_phase_async(map.clone(), p.clone(), i, start.clone(), &mut seq, &thread);
    }
}

// }}} bencher

// {{{ cli

pub fn cli() {
    env_logger::init();

    #[derive(Parser, Debug)]
    #[command(about)]
    struct Args {
        #[arg(long, short = 'f')]
        file: Option<String>,

        #[arg(long, short = 'm')]
        map_file: Option<String>,

        #[arg(long, short = 'b')]
        benchmark_file: Option<String>,
    }

    let args = Args::parse();
    debug!("Starting benchmark with args: {:?}", args);

    let opt: String = if let Some(f) = args.file {
        read_to_string(f.as_str()).unwrap()
    } else {
        let m = args.map_file.clone().unwrap();
        let b = args.benchmark_file.clone().unwrap();
        read_to_string(m.as_str()).unwrap() + "\n" + &read_to_string(b.as_str()).unwrap()
    };
    let (map, phases) = init(&opt);
    map.bench(&phases);
}

// }}} cli

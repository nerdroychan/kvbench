//! The core benchmark functionality.
//!
//! A benchmark in this crate actually refers to a group of benchmark runs, named **phases**. Users
//! can provide one or multiple phases that will be run sequentially, each with different
//! configurations.
//!
//! ## Configuration Format
//!
//! A benchmark configuration file is formatted in TOML. It consists of the definition of multiple
//! phases, each is defined in a dictionary named `benchmark`. Phases are organized in an array, so
//! the configuration of each phase starts with `[[benchmark]]`. It also supports a `[global]`
//! section in the configuration file that will override the missing field in each phase. This can
//! reduce the number of repeated options in each phase (e.g., shared options).
//!
//! A configuration file generally looks like the following:
//!
//! ```toml
//! [global]
//! # global options
//!
//! [[benchmark]]
//! # phase 1 configuration
//!
//! [[benchmark]]
//! # phase 2 configuration
//!
//! ...
//! ```
//!
//! Available options and their usage can be found in [`BenchmarkOpt`] and [`GlobalOpt`], for phase
//! and global options, respectively.
//!
//! Options in `[global]` section can be overwritten via environment variables without changing the
//! content in the TOML file.
//! For example, if the user needs to override `x` in `[global]`, setting the environment variable
//! `global.x` will get the job done.
//!
//! ## Output Format
//!
//! When measuring throughput, an output may look like the following:
//! ```txt
//! 0 phase 0 repeat 0 duration 1.00 elapsed 1.00 total 1000000 mops 1.00
//! 1 phase 0 repeat 1 duration 1.00 elapsed 2.00 total 1000000 mops 1.00
//! 2 phase 0 repeat 2 duration 1.00 elapsed 3.00 total 1000000 mops 1.00
//! 3 phase 0 finish . duration 1.00 elapsed 3.00 total 3000000 mops 1.00
//! ```
//!
//! From the first element to the last element in each line, the meanings are:
//! - Report sequence number.
//! - "phase" followed by the phase id.
//! - "repeat" followed by the repeat id in a phase, or "finish .", if it is the aggregated report
//! of a whole phase.
//! - "duration" followed by the duration of the repeat/phase, in seconds.
//! - "elapsed" followed by the total elapsed time when this line is printed, since the starting of
//! all the benchmarks.
//! - "total" followed by the total key-value operations executed by all worker threads in the
//! repeat/phase.
//! - "mops" followed by the throughput in million operations per second of the repeat/phase.

use crate::stores::{BenchKVMap, BenchKVMapOpt};
use crate::thread::{JoinHandle, Thread};
use crate::workload::{Workload, WorkloadOpt};
use crate::*;
use figment::providers::{Env, Format, Toml};
use figment::Figment;
use hashbrown::hash_map::HashMap;
use hdrhistogram::Histogram;
use log::debug;
use parking_lot::Mutex;
use quanta::Instant;
use serde::Deserialize;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

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

/// The configuration of a single benchmark deserialized from a TOML string.
///
/// The fields are optional to ease parsing from TOML, as there can be global parameters that are
/// set for them.
#[derive(Deserialize, Clone, Debug)]
pub struct BenchmarkOpt {
    /// Number of threads that runs this benchmark.
    ///
    /// Default: 1.
    pub threads: Option<usize>,

    /// How many times this benchmark will be repeated.
    ///
    /// Default: 1.
    pub repeat: Option<usize>,

    /// How long this benchmark will run, unit is seconds. If this option is specified, the `ops`
    /// option will be ignored.
    ///
    /// Note: see `ops`.
    pub timeout: Option<f32>,

    /// How many operations each worker will execute. Only used if `timeout` is not given.
    ///
    /// Note: if both `timeout` and `ops` are not given, the run is only stopped when all possible
    /// keys are generated.
    pub ops: Option<u64>,

    /// Report mode:
    ///
    /// - "hidden": not reported.
    /// - "repeat": after each repeat, the metrics for that repeat is printed.
    /// - "finish": after all repeats are finished, the metrics of the whole phase is printed.
    /// - "all": equals to "repeat" + "finish".
    pub report: Option<String>,

    /// Max depth of queue for each worker. Only useful with [`AsyncKVMap`].
    ///
    /// When the pending requests are less than `qd`, the worker will not attempt to get more
    /// responses.
    ///
    /// Default: 1.
    pub qd: Option<usize>,

    /// Batch size for each request Only useful with [`AsyncKVMap`].
    ///
    /// Default: 1.
    pub batch: Option<usize>,

    /// Whether or not to record latency during operation. Since measuring time is of extra cost,
    /// enabling latency measurement usually affects the throughput metrics.
    ///
    /// Default: false.
    pub latency: Option<bool>,

    /// The definition of a workload.
    ///
    /// This section is embedded and flattened, so that you can directly use options in
    /// [`WorkloadOpt`].
    #[serde(flatten)]
    pub workload: WorkloadOpt,
}

impl BenchmarkOpt {
    /// Internal function called after all global options are applied and when all the options are
    /// set. This will test if the opt can be a valid benchmark. It does not check the workload's
    /// configuration, as it will be checked when a workload instance is created.
    fn sanity(&self) {
        // these must be present, so `unwrap` won't panic.
        assert!(
            *self.threads.as_ref().unwrap() > 0,
            "threads should be positive if given"
        );
        assert!(
            *self.repeat.as_ref().unwrap() > 0,
            "repeat should be positive if given"
        );
        match self.report.as_ref().unwrap().as_str() {
            "hidden" | "repeat" | "finish" | "all" => {}
            _ => panic!("report mode should be one of: hidden, repeat, finish, all"),
        }
        assert!(
            *self.qd.as_ref().unwrap() > 0,
            "queue depth should be positive if given"
        );
        assert!(
            *self.batch.as_ref().unwrap() > 0,
            "queue depth should be positive if given"
        );
    }
}

/// The configuration of a benchmark, parsed from user's input.
#[derive(Debug)]
pub struct Benchmark {
    threads: usize,
    repeat: usize,
    qd: usize,
    batch: usize,
    len: Length,
    report: ReportMode,
    latency: bool,
    wopt: WorkloadOpt,
}

const TIME_CHECK_INTERVAL: u64 = 32;

impl Benchmark {
    /// The constructor of Benchmark expects all fields have their values, the struct should
    /// contain either its own parameters, or carry the default parameters.
    fn new(opt: &BenchmarkOpt) -> Self {
        opt.sanity();
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
            _ => panic!("Invalid report mode provided"),
        };
        let latency = opt.latency.unwrap();
        let wopt = opt.workload.clone();
        Self {
            threads,
            repeat,
            qd,
            batch,
            len,
            report,
            latency,
            wopt,
        }
    }
}

// }}} benchmark

// {{{ benchmarkgroup

/// The global options that go to the `[global]` section.
///
/// They will override missing fields in each `[[benchmark]]` section, if the corresponding option
/// is missing. For the usage of each option, please refer to [`BenchmarkOpt`].
#[derive(Deserialize, Clone, Debug)]
pub struct GlobalOpt {
    pub threads: Option<usize>,
    pub repeat: Option<usize>,
    pub qd: Option<usize>,
    pub batch: Option<usize>,
    pub report: Option<String>,
    pub latency: Option<bool>,
    pub klen: Option<usize>,
    pub vlen: Option<usize>,
    pub kmin: Option<usize>,
    pub kmax: Option<usize>,
}

impl Default for GlobalOpt {
    fn default() -> Self {
        Self {
            threads: None,
            repeat: None,
            qd: None,
            batch: None,
            report: None,
            latency: None,
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
        opt.latency = opt
            .latency
            .clone()
            .or_else(|| Some(self.latency.clone().unwrap_or(false)));
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
    let opt: BenchmarkGroupOpt = Figment::new()
        .merge(Toml::string(text))
        .merge(Env::raw())
        .extract()
        .unwrap();
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

fn bench_phase_should_break(
    len: &Length,
    count: u64,
    start: &Instant,
    workload: &mut Workload,
) -> bool {
    match len {
        Length::Count(c) => {
            if count == *c {
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

/// A per-worker counter for each repeat in the same benchmark. Using [`AtomicU64`] here makes the
/// measurement `Sync` + `Send` so it can be freely accessed by different threads (mainly by the
/// thread that aggregates the overall measurement).
struct Counter(AtomicU64);

impl Counter {
    fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    fn read(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    fn reference(&self) -> &mut u64 {
        // SAFETY: the counter() method will only be called by the thread that updates its value
        unsafe { &mut *self.0.as_ptr() }
    }
}

/// A per-worker latency collector for each repeat in the same benchmark. This is only accessed and
/// collected at the end of each benchmark.
struct Latency {
    /// Async only: request_id -> submit time
    pending: HashMap<usize, Instant>,

    /// Latency histogram in us, maximum latency recorded here is 1 second
    hdr: Histogram<u64>,
}

impl Latency {
    fn new() -> Self {
        let pending = HashMap::new();
        let hdr = Histogram::new(3).unwrap();
        Self { pending, hdr }
    }

    fn record(&mut self, duration: Duration) {
        let us = duration.as_nanos() as u64;
        assert!(self.hdr.record(us).is_ok());
    }

    fn async_register(&mut self, id: usize, t: Instant) {
        self.pending.insert(id, t);
    }

    fn async_record(&mut self, id: usize, t: Instant) {
        let d = t - self.pending.remove(&id).unwrap();
        self.record(d);
    }

    fn merge(&mut self, other: &Latency) {
        assert!(self.pending.is_empty() && other.pending.is_empty());
        assert!(self.hdr.add(&other.hdr).is_ok());
    }
}

/// The main metrics for each worker thread in the same benchmark.
struct Measurement {
    /// Per-repeat counters. This value is actively updated by the worker and loosely evaluated by
    /// the main thread.
    counters: Vec<Counter>,

    /// Per-worker latency metrics. This value is avtively updated by the worker if latency needs
    /// to be checked, and is shared among all repeats. It is only merged at the end of a whole
    /// benchmark.
    latency: Mutex<Latency>,

    /// The duration of each repeat that is measured by the corresponding worker thread. It is only
    /// updated once after a repeat is really done. In a time-limited run, the master thread will
    /// try to access the duration. If an entry exists, it means the thread has finished execution,
    /// so the master will directly use the time duration observed by the worker. If an entry is
    /// not here, the time will be observed by the master.
    durations: Vec<Mutex<Option<Duration>>>,
}

impl Measurement {
    fn new(repeat: usize) -> Self {
        let counters = (0..repeat).into_iter().map(|_| Counter::new()).collect();
        let latency = Mutex::new(Latency::new());
        let durations = (0..repeat).into_iter().map(|_| Mutex::new(None)).collect();
        Self {
            counters,
            latency,
            durations,
        }
    }
}

struct WorkerContext {
    /// The benchmark phase that the current work is referring to
    benchmark: Arc<Benchmark>,

    /// The very beginning of all benchmarks in a group, for calculating elapsed timestamp
    since: Instant,

    /// The current phase of this benchmark in the group
    phase: usize,

    /// The measurement of all worker threads. One worker typically only needs to refer to one of
    /// them, and the master thread (thread.id == repeat) will aggregate the metrics and make an
    /// output
    measurements: Vec<Arc<Measurement>>,

    /// The sequence number of the output. This value is actually only used by one thread at a
    /// time, but it is shared among all workers.
    seq: Arc<AtomicUsize>,

    /// Barrier that syncs all workers
    barrier: Arc<Barrier>,

    /// `(worker_id, nr_threads)` pair, used to determine the identity of a worker and also
    thread_info: (usize, usize),
}

fn bench_stat_repeat(
    benchmark: &Arc<Benchmark>,
    phase: usize,
    repeat: usize,
    since: Instant,
    start: Instant,
    end: Instant,
    thread_info: (usize, usize),
    seq: &Arc<AtomicUsize>,
    measurements: &Vec<Arc<Measurement>>,
) {
    assert!(thread_info.0 == 0);
    let mut throughput = 0.0f64;
    let mut total = 0u64;
    for i in 0..thread_info.1 {
        let d = match *measurements[i].durations[repeat].lock() {
            Some(d) => d,
            None => {
                // only applies to time-limited benchmarks
                assert!(matches!(benchmark.len, Length::Timeout(_)));
                start.elapsed()
            }
        };
        let ops = measurements[i].counters[repeat].read();
        let tput = ops as f64 / d.as_secs_f64() / 1_000_000.0;
        total += ops;
        throughput += tput;
    }

    let duration = (end - start).as_secs_f64();
    let elapsed = (end - since).as_secs_f64();

    if benchmark.report == ReportMode::Repeat || benchmark.report == ReportMode::All {
        println!(
            "{} phase {} repeat {} duration {:.2} elapsed {:.2} total {} mops {:.2}",
            seq.load(Ordering::Relaxed),
            phase,
            repeat,
            duration,
            elapsed,
            total,
            throughput,
        );
        seq.fetch_add(1, Ordering::Relaxed);
    }
}

fn bench_stat_final(
    benchmark: &Arc<Benchmark>,
    phase: usize,
    since: Instant,
    start: Instant,
    end: Instant,
    thread_info: (usize, usize),
    seq: &Arc<AtomicUsize>,
    measurements: &Vec<Arc<Measurement>>,
) {
    assert!(thread_info.0 == 0);
    let mut total = 0u64;
    let mut latency = Latency::new();
    for i in 0..thread_info.1 {
        for j in 0..benchmark.repeat {
            let ops = measurements[i].counters[j].read();
            total += ops;
        }
        latency.merge(&measurements[i].latency.lock());
    }

    let duration = (end - start).as_secs_f64();
    let elapsed = (end - since).as_secs_f64();

    let throughput = total as f64 / duration / 1_000_000.0;

    if benchmark.report == ReportMode::Finish || benchmark.report == ReportMode::All {
        print!(
            "{} phase {} finish . duration {:.2} elapsed {:.2} total {} mops {:.2}",
            seq.load(Ordering::Relaxed),
            phase,
            duration,
            elapsed,
            total,
            throughput,
        );
        if benchmark.latency {
            print!(" ");
            assert_eq!(total, latency.hdr.len());
            let hdr = &latency.hdr;
            print!(
                "min_ns {} max_ns {} p50_ns {} p95_ns {} p99_ns {} p999_ns {}",
                hdr.min(),
                hdr.max(),
                hdr.value_at_quantile(0.50),
                hdr.value_at_quantile(0.95),
                hdr.value_at_quantile(0.99),
                hdr.value_at_quantile(0.999),
            );
        }
        println!();
        seq.fetch_add(1, Ordering::Relaxed);
    }
}

fn bench_worker_regular(
    map: Arc<Box<impl KVMap + ?Sized>>,
    context: WorkerContext,
    thread: impl Thread,
) {
    let WorkerContext {
        benchmark,
        since,
        phase,
        measurements,
        seq,
        barrier,
        thread_info,
    } = context;

    let id = thread_info.0;
    thread.pin(id);

    // if record latency, take the lock guard of the latency counter until all repeats are done
    let mut latency = match benchmark.latency {
        true => Some(measurements[id].latency.lock()),
        false => None,
    };

    let latency_tick = match latency {
        Some(_) => || Some(Instant::now()),
        None => || None,
    };

    let mut handle = map.handle();
    let mut rng = rand::thread_rng();
    let mut workload = Workload::new(&benchmark.wopt, Some(thread_info));
    let start = Instant::now(); // for thread 0
    for i in 0..benchmark.repeat {
        let counter = measurements[id].counters[i].reference();
        // start the benchmark phase at roughly the same time
        barrier.wait();
        let start = Instant::now();
        // start benchmark
        loop {
            let op = workload.next(&mut rng);
            let op_start = latency_tick();
            match op {
                Operation::Set { key, value } => {
                    handle.set(&key[..], &value[..]);
                }
                Operation::Get { key } => {
                    let _ = handle.get(&key[..]);
                }
                Operation::Delete { key } => {
                    handle.delete(&key[..]);
                }
            }
            let op_end = latency_tick();
            if let Some(ref mut l) = latency {
                l.record(op_end.unwrap() - op_start.unwrap());
            }
            *counter += 1;
            // check if we need to break
            if bench_phase_should_break(&benchmark.len, *counter, &start, &mut workload) {
                workload.reset();
                break;
            }
        }

        // after the execution, counter is up-to-date, so it's time to update duration
        let end = Instant::now();
        *measurements[id].durations[i].lock() = Some(end.duration_since(start.clone()));

        // for non time-limited benchmarks, sync first to make sure that all threads have finished
        // if a benchmark is time limited, loosely evaluate the metrics
        if !matches!(benchmark.len, Length::Timeout(_)) {
            barrier.wait();
        }

        // master is 0, it will aggregate data and print info inside this call
        if id == 0 {
            bench_stat_repeat(
                &benchmark,
                phase,
                i,
                since,
                start,
                end,
                thread_info,
                &seq,
                &measurements,
            );
        }
    }

    drop(latency);

    // every thread will sync on this
    barrier.wait();

    if id == 0 {
        let end = Instant::now();
        bench_stat_final(
            &benchmark,
            phase,
            since,
            start,
            end,
            thread_info,
            &seq,
            &measurements,
        );
    }
}

fn bench_worker_async(
    map: Arc<Box<impl AsyncKVMap + ?Sized>>,
    context: WorkerContext,
    thread: impl Thread,
) {
    let WorkerContext {
        benchmark,
        since,
        phase,
        measurements,
        seq,
        barrier,
        thread_info,
    } = context;

    let id = thread_info.0;
    thread.pin(id);

    // if record latency, take the lock guard of the latency counter until all repeats are done
    let mut latency = match benchmark.latency {
        true => Some(measurements[id].latency.lock()),
        false => None,
    };

    let responder = Rc::new(RefCell::new(Vec::<Response>::new()));
    let mut handle = map.handle(responder.clone());
    let mut rng = rand::thread_rng();
    let mut workload = Workload::new(&benchmark.wopt, Some(thread_info));
    // pending requests is global, as it is not needed to drain all requests after each repeat
    let mut pending = 0usize;
    let mut requests = Vec::<Request>::with_capacity(benchmark.batch);
    let mut rid = 0usize;
    let start = Instant::now(); // for thread 0
    for i in 0..benchmark.repeat {
        let counter = measurements[id].counters[i].reference();
        // start the benchmark phase at roughly the same time
        barrier.wait();
        let start = Instant::now();
        // start benchmark
        loop {
            // first clear the requests vector
            requests.clear();
            // sample requests
            for _ in 0..benchmark.batch {
                let op = workload.next(&mut rng);
                requests.push(Request { id: rid, op });
                rid += 1;
                // need to add to count here, instead of after this loop
                // otherwise the last check may fail because the time check is after a certain
                // interval, but the mod is never 0
                *counter += 1;
                if bench_phase_should_break(&benchmark.len, *counter, &start, &mut workload) {
                    break;
                }
            }
            // now we have a batch, send it all, whatever its size is
            let len = requests.len();
            handle.submit(&requests);
            pending += len;
            if let Some(ref mut l) = latency {
                let submit = Instant::now();
                for r in requests.iter() {
                    l.async_register(r.id, submit);
                }
            }
            if bench_phase_should_break(&benchmark.len, *counter, &start, &mut workload) {
                workload.reset();
                break;
            }
            // use a loop to make sure that pending is under qd, only drain the handle if the bench
            // phase is not ending
            loop {
                handle.drain();
                let responses = responder.replace_with(|_| Vec::new());
                pending -= responses.len();
                if let Some(ref mut l) = latency {
                    let submit = Instant::now();
                    for r in responses.iter() {
                        l.async_record(r.id, submit);
                    }
                }
                if pending <= benchmark.qd {
                    break;
                }
            }
        }

        // after the execution, counter is up-to-date, so it's time to update duration
        let end = Instant::now();
        *measurements[id].durations[i].lock() = Some(end.duration_since(start.clone()));

        // for non time-limited benchmarks, sync first to make sure that all threads have finished
        // if a benchmark is time limited, loosely evaluate the metrics
        if !matches!(benchmark.len, Length::Timeout(_)) {
            barrier.wait();
        }

        // master is 0, it will aggregate data and print info inside this call
        if id == 0 {
            bench_stat_repeat(
                &benchmark,
                phase,
                i,
                since,
                start,
                end,
                thread_info,
                &seq,
                &measurements,
            );
        }
    }

    // wait until all requests are back
    loop {
        if pending == 0 {
            break;
        }
        handle.drain();
        let responses = responder.replace_with(|_| Vec::new());
        pending -= responses.len();
        if let Some(ref mut l) = latency {
            let submit = Instant::now();
            for r in responses.iter() {
                l.async_record(r.id, submit);
            }
        }
    }

    drop(latency);

    // every thread will sync on this
    barrier.wait();

    if id == 0 {
        let end = Instant::now();
        bench_stat_final(
            &benchmark,
            phase,
            since,
            start,
            end,
            thread_info,
            &seq,
            &measurements,
        );
    }
}

fn bench_phase_regular(
    map: Arc<Box<impl KVMap + ?Sized>>,
    benchmark: Arc<Benchmark>,
    phase: usize,
    since: Arc<Instant>,
    seq: Arc<AtomicUsize>,
    thread: &impl Thread,
) {
    let barrier = Arc::new(Barrier::new(benchmark.threads.try_into().unwrap()));
    let measurements: Vec<Arc<Measurement>> = (0..benchmark.threads)
        .map(|_| Arc::new(Measurement::new(benchmark.repeat)))
        .collect();
    let mut handles = Vec::new();
    for t in 0..benchmark.threads {
        let map = map.clone();
        let benchmark = benchmark.clone();
        let barrier = barrier.clone();
        let thread_info = (t, benchmark.threads);
        let context = WorkerContext {
            benchmark,
            phase,
            measurements: measurements.clone(),
            barrier,
            seq: seq.clone(),
            since: *since,
            thread_info,
        };
        let worker_thread = thread.clone();
        let handle = thread.spawn(move || {
            bench_worker_regular(map, context, worker_thread);
        });
        handles.push(handle);
    }

    // join thread 0
    handles.pop().unwrap().join();

    while let Some(handle) = handles.pop() {
        handle.join();
    }
}

fn bench_phase_async(
    map: Arc<Box<impl AsyncKVMap + ?Sized>>,
    benchmark: Arc<Benchmark>,
    phase: usize,
    since: Arc<Instant>,
    seq: Arc<AtomicUsize>,
    thread: &impl Thread,
) {
    let barrier = Arc::new(Barrier::new((benchmark.threads).try_into().unwrap()));
    let measurements: Vec<Arc<Measurement>> = (0..benchmark.threads)
        .map(|_| Arc::new(Measurement::new(benchmark.repeat)))
        .collect();
    let mut handles = Vec::new();
    for t in 0..benchmark.threads {
        let map = map.clone();
        let benchmark = benchmark.clone();
        let barrier = barrier.clone();
        let thread_info = (t, benchmark.threads);
        let context = WorkerContext {
            benchmark,
            phase,
            measurements: measurements.clone(),
            barrier,
            seq: seq.clone(),
            since: *since,
            thread_info,
        };
        let worker_thread = thread.clone();
        let handle = thread.spawn(move || {
            bench_worker_async(map, context, worker_thread);
        });
        handles.push(handle);
    }

    handles.pop().unwrap().join();

    while let Some(handle) = handles.pop() {
        handle.join();
    }
}

/// The real benchmark function for [`KVMap`].
///
/// **You may not need to check this if it is OK to run benchmarks with [`std::thread`].**
pub fn bench_regular(
    map: Arc<Box<impl KVMap + ?Sized>>,
    phases: &Vec<Arc<Benchmark>>,
    thread: impl Thread,
) {
    debug!("Running regular bencher");
    let start = Arc::new(Instant::now());
    let seq = Arc::new(AtomicUsize::new(0));
    for (i, p) in phases.iter().enumerate() {
        bench_phase_regular(
            map.clone(),
            p.clone(),
            i,
            start.clone(),
            seq.clone(),
            &thread,
        );
    }
}

/// The real benchmark function for [`AsyncKVMap`].
///
/// **You may not need to check this if it is OK to run benchmarks with [`std::thread`].**
pub fn bench_async(
    map: Arc<Box<impl AsyncKVMap + ?Sized>>,
    phases: &Vec<Arc<Benchmark>>,
    thread: impl Thread,
) {
    debug!("Running async bencher");
    let start = Arc::new(Instant::now());
    let seq = Arc::new(AtomicUsize::new(0));
    for (i, p) in phases.iter().enumerate() {
        bench_phase_async(
            map.clone(),
            p.clone(),
            i,
            start.clone(),
            seq.clone(),
            &thread,
        );
    }
}

// }}} bencher

// {{{ tests

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE_BENCH: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/presets/benchmarks/example.toml"
    ));

    fn example(map_opt: &str) {
        let _ = env_logger::try_init();
        let opt = map_opt.to_string() + "\n" + EXAMPLE_BENCH;
        let (map, phases) = init(&opt);
        map.bench(&phases);
    }

    #[test]
    fn example_null() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/null.toml"
        ));
        example(OPT);
    }

    #[test]
    fn example_null_async() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/null_async.toml"
        ));
        example(OPT);
    }

    #[test]
    fn example_mutex_hashmap() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/mutex_hashmap.toml"
        ));
        example(OPT);
    }

    #[test]
    fn example_rwlock_hashmap() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/rwlock_hashmap.toml"
        ));
        example(OPT);
    }

    #[test]
    #[cfg(feature = "dashmap")]
    fn example_dashmap() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/dashmap.toml"
        ));
        example(OPT);
    }

    #[test]
    #[cfg(feature = "contrie")]
    fn example_contrie() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/contrie.toml"
        ));
        example(OPT);
    }

    #[test]
    #[cfg(feature = "chashmap")]
    fn example_chashmap() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/chashmap.toml"
        ));
        example(OPT);
    }

    #[test]
    #[cfg(feature = "scc")]
    fn example_scchashmap() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/scchashmap.toml"
        ));
        example(OPT);
    }

    #[test]
    #[cfg(feature = "flurry")]
    fn example_flurry() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/flurry.toml"
        ));
        example(OPT);
    }

    #[test]
    #[cfg(feature = "papaya")]
    fn example_papaya() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/papaya.toml"
        ));
        example(OPT);
    }

    #[test]
    fn example_mutex_btreemap() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/mutex_btreemap.toml"
        ));
        example(OPT);
    }

    #[test]
    fn example_rwlock_btreemap() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/rwlock_btreemap.toml"
        ));
        example(OPT);
    }
}

// }}} tests

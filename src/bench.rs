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
//! Currently, all outputs are in plain text format. This makes the output easy to process using
//! shell scripts and tools including gnuplot. If there are new data added to the output, it
//! will be appended at the end of existing entries (but before `cdf` if it exists, see below)
//! to make sure outputs from old versions can still be processed without changes.
//!
//! ### Throughput-only Output (default case)
//!
//! When measuring throughput, an output may look like the following:
//! ```txt
//! phase 0 repeat 0 duration 1.00 elapsed 1.00 total 1000000 mops 1.00
//! phase 0 repeat 1 duration 1.00 elapsed 2.00 total 1000000 mops 1.00
//! phase 0 repeat 2 duration 1.00 elapsed 3.00 total 1000000 mops 1.00
//! phase 0 finish . duration 1.00 elapsed 3.00 total 3000000 mops 1.00
//! ```
//!
//! The general format is:
//!
//! ```txt
//! phase <p> repeat <r> duration <d> elapsed <e> total <o> mops <t>
//! ```
//!
//! Where:
//!
//! - `<p>`: phase id.
//! - `<r>`: repeat id in a phase, or string `finish .`, if the line is the aggregated report
//! of a whole phase.
//! - `<d>`: the duration of the repeat/phase, in seconds.
//! - `<e>`: the total elapsed seconds since the starting of the program.
//! - `<o>`: the total key-value operations executed by all worker threads in the repeat/phase.
//! - `<t>`: followed by the throughput in million operations per second of the repeat/phase.
//!
//! ### Throughput + Latency Output (when `latency` is `true`)
//!
//! When latency measurement is enabled, the latency metrics shall be printed at the end of each
//! benchmark. It is not shown after each repeat, because unlike throughput which is a singleton
//! value at a given time, latency is a set of values and it usually matters only when we aggregate
//! a lot of them. The output format in this case is generally the same as throughput-only
//! measurements, but the `finish` line has extra output like the following:
//!
//! ```txt
//! phase 0 repeat 0 duration 1.00 elapsed 1.00 total 1000000 mops 1.00
//! phase 0 repeat 1 duration 1.00 elapsed 2.00 total 1000000 mops 1.00
//! phase 0 repeat 2 duration 1.00 elapsed 3.00 total 1000000 mops 1.00
//! phase 0 finish . duration 1.00 elapsed 3.00 total 3000000 mops 1.00 min_us 0.05 max_us 100.00 avg_us 50.00 p50_us 50.00 p95_us 95.00 p99_us 99.00 p999_us 100.00
//! ```
//!
//! The extra output on the last line has a format of:
//!
//! ```txt
//! min_us <i> max_us <a> avg_us <v> p50_us <m> p95_us <n> p99_us <p> p999_us <t>
//! ```
//!
//! Where (all units are microseconds):
//!
//! - `<i>`: minimum latency
//! - `<a>`: maximum latency
//! - `<v>`: mean latency
//! - `<m>`: median latency (50% percentile)
//! - `<n>`: P95 latency
//! - `<p>`: P99 latency
//! - `<t>`: P999 latency (99.9%)
//!
//! ### Throughput + Latency + Latency CDF Mode (when both `latency` and `cdf` are `true`)
//!
//! When `cdf` is enabled, the latency CDF data will be printed at the end of the same line as the
//! latency metrics above. In that case, the output will be like the following:
//!
//! ```txt
//! phase 0 repeat 0 duration 1.00 elapsed 1.00 total 1000000 mops 1.00
//! phase 0 repeat 1 duration 1.00 elapsed 2.00 total 1000000 mops 1.00
//! phase 0 repeat 2 duration 1.00 elapsed 3.00 total 1000000 mops 1.00
//! phase 0 finish . duration 1.00 elapsed 3.00 total 3000000 mops 1.00 min_us 0.05 max_us 100.00 avg_us 50.00 p50_us 50.00 p95_us 95.00 p99_us 99.00 p999_us 100.00 cdf_us percentile ...
//! ```
//! Since the latency metrics vary a lot between different benchmarks/runs, the number of data
//! points of the CDF is different. Therefore, it is printed at the end of the output only. It is
//! printed as a tuple of `<us> <percentile>` where `<us>` is the latency in microseconds and
//! `<percentile>` is the percentile of the accumulated operations with latency higher than between
//! `<ns> - 1` and `<ns>`, inclusively, ranging from 0 to 100 (two digit precision).
//! There can be arbitrary number of tuples. The output ends when the maximum recorded latency is
//! reached.
//!
//! An example of the CDF data will look like:
//!
//! ```txt
//! cdf_us percentile 1 0.00 2 0.00 3 0.00 4 10.00 5 20.00 6 20.00 ...
//! ```
//!
//! It means there are not data points at 1/2/3 microseconds. At 4 microseconds, there are 10% data
//! points. At 5 microseconds, there are another 10% data points which makes the total percentile
//! 20.00. At 6 microseconds, there are no data points so the percentile is still 20.00. Users can
//! post-process the output and make a smooth CDF plot out of it.

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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

// {{{ benchmark

/// Length determines when a benchmark should stop or how often the metrics should be collected.
#[derive(Clone, Debug, PartialEq)]
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

    /// How many times this benchmark will be repeated. This option is useful when user would like
    /// to plot the performance trend over time in the same benchmark. For example, setting this
    /// option to 100 with one second timeout for each repeat can provide 100 data points over a
    /// 100 second period.
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

    /// Whether or not to print out latency CDF at the end of each benchmark. If this is set to
    /// `true`, `latency` must also be set to `true`.
    ///
    /// Default: false.
    pub cdf: Option<bool>,

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
    ///
    /// Note: `timeout` and `ops` may not be set as of now, and they are not checked. They will be
    /// converted to `Length` when creating a new benchmark object.
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
        if let Some(true) = self.cdf {
            assert!(
                *self.latency.as_ref().unwrap(),
                "when cdf is true, latency must also be true"
            );
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
#[derive(Debug, PartialEq)]
pub struct Benchmark {
    threads: usize,
    repeat: usize,
    qd: usize,
    batch: usize,
    len: Length,
    report: ReportMode,
    latency: bool,
    cdf: bool,
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
            assert!(
                opt.ops.is_none(),
                "timeout and ops cannot be provided at the same time"
            );
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
        let cdf = opt.cdf.unwrap();
        let wopt = opt.workload.clone();
        Self {
            threads,
            repeat,
            qd,
            batch,
            len,
            report,
            latency,
            cdf,
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
    // benchmark
    pub threads: Option<usize>,
    pub repeat: Option<usize>,
    pub qd: Option<usize>,
    pub batch: Option<usize>,
    pub report: Option<String>,
    pub latency: Option<bool>,
    pub cdf: Option<bool>,
    // workload
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
            cdf: None,
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
        opt.cdf = opt
            .cdf
            .clone()
            .or_else(|| Some(self.cdf.clone().unwrap_or(false)));
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
            "phase {} repeat {} duration {:.2} elapsed {:.2} total {} mops {:.2}",
            phase, repeat, duration, elapsed, total, throughput,
        );
    }
}

fn bench_stat_final(
    benchmark: &Arc<Benchmark>,
    phase: usize,
    since: Instant,
    start: Instant,
    end: Instant,
    thread_info: (usize, usize),
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
            "phase {} finish . duration {:.2} elapsed {:.2} total {} mops {:.2}",
            phase, duration, elapsed, total, throughput,
        );
        if benchmark.latency {
            print!(" ");
            assert_eq!(total, latency.hdr.len());
            let hdr = &latency.hdr;
            print!(
                "min_us {:.2} max_us {:.2} avg_us {:.2} \
                 p50_us {:.2} p95_us {:.2} p99_us {:.2} p999_us {:.2}",
                hdr.min() as f64 / 1000.0,
                hdr.max() as f64 / 1000.0,
                hdr.mean() / 1000.0,
                hdr.value_at_quantile(0.50) as f64 / 1000.0,
                hdr.value_at_quantile(0.95) as f64 / 1000.0,
                hdr.value_at_quantile(0.99) as f64 / 1000.0,
                hdr.value_at_quantile(0.999) as f64 / 1000.0,
            );
            if benchmark.cdf {
                print!(" cdf_us percentile ");
                let mut cdf = 0;
                for v in latency.hdr.iter_linear(1000) {
                    let ns = v.value_iterated_to();
                    let us = (ns + 1) / 1000;
                    cdf += v.count_since_last_iteration();
                    print!("{} {:.2}", us, cdf as f64 * 100.0 / total as f64);
                    if ns >= hdr.max() {
                        break;
                    }
                    print!(" ");
                }
                assert_eq!(cdf, total);
            }
        }

        println!();
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
                Operation::Scan { key, n } => {
                    let _ = handle.scan(&key[..], n);
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
            &measurements,
        );
    }
}

fn bench_phase_regular(
    map: Arc<Box<impl KVMap + ?Sized>>,
    benchmark: Arc<Benchmark>,
    phase: usize,
    since: Arc<Instant>,
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
    for (i, p) in phases.iter().enumerate() {
        bench_phase_regular(map.clone(), p.clone(), i, start.clone(), &thread);
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
    for (i, p) in phases.iter().enumerate() {
        bench_phase_async(map.clone(), p.clone(), i, start.clone(), &thread);
    }
}

// }}} bencher

// {{{ tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_options_are_applied() {
        let opt = r#"
            [map]
            name = "nullmap"

            [global]
            threads = 8
            repeat = 10
            qd = 10
            batch = 15
            report = "finish"
            latency = true
            cdf = true
            klen = 8
            vlen = 16
            kmin = 100
            kmax = 1000

            [[benchmark]]
            timeout = 10.0
            set_perc = 50
            get_perc = 30
            del_perc = 10
            scan_perc = 10
            dist = "incrementp"
        "#;

        let (_, bg) = init(opt);
        assert_eq!(bg.len(), 1);

        let wopt = WorkloadOpt {
            set_perc: Some(50),
            get_perc: Some(30),
            del_perc: Some(10),
            scan_perc: Some(10),
            dist: "incrementp".to_string(),
            scan_n: None,
            klen: Some(8),
            vlen: Some(16),
            kmin: Some(100),
            kmax: Some(1000),
            zipf_theta: None,
            zipf_hotspot: None,
        };

        let benchmark = Benchmark {
            threads: 8,
            repeat: 10,
            qd: 10,
            batch: 15,
            report: ReportMode::Finish,
            latency: true,
            cdf: true,
            len: Length::Timeout(Duration::from_secs_f32(10.0)),
            wopt,
        };

        assert_eq!(*bg[0], benchmark)
    }

    #[test]
    fn global_options_defaults_are_applied() {
        let opt = r#"
            [map]
            name = "nullmap"

            [[benchmark]]
            set_perc = 50
            get_perc = 30
            del_perc = 10
            scan_perc = 10
            klen = 8
            vlen = 16
            kmin = 1
            kmax = 1000
            dist = "shufflep"
        "#;

        let (_, bg) = init(opt);
        assert_eq!(bg.len(), 1);

        let wopt = WorkloadOpt {
            set_perc: Some(50),
            get_perc: Some(30),
            del_perc: Some(10),
            scan_perc: Some(10),
            scan_n: None,
            dist: "shufflep".to_string(),
            klen: Some(8),
            vlen: Some(16),
            kmin: Some(1),
            kmax: Some(1000),
            zipf_theta: None,
            zipf_hotspot: None,
        };

        let benchmark = Benchmark {
            threads: 1,
            repeat: 1,
            qd: 1,
            batch: 1,
            report: ReportMode::All,
            latency: false,
            cdf: false,
            len: Length::Exhaust,
            wopt,
        };

        assert_eq!(*bg[0], benchmark)
    }

    #[test]
    #[should_panic(expected = "should be positive")]
    fn invalid_threads() {
        let opt = r#"
            [map]
            name = "nullmap"

            [global]
            klen = 8
            vlen = 16
            kmin = 0
            kmax = 1000

            [[benchmark]]
            threads = 0
            timeout = 1.0
            set_perc = 100
            get_perc = 0
            del_perc = 0
            scan_perc = 0
            dist = "incrementp"
        "#;

        let (_, _) = init(opt);
    }

    #[test]
    #[should_panic(expected = "should be positive")]
    fn invalid_repeat() {
        let opt = r#"
            [map]
            name = "nullmap"

            [global]
            klen = 8
            vlen = 16
            kmin = 0
            kmax = 1000

            [[benchmark]]
            repeat = 0
            timeout = 1.0
            set_perc = 100
            get_perc = 0
            del_perc = 0
            scan_perc = 0
            dist = "incrementp"
        "#;

        let (_, _) = init(opt);
    }

    #[test]
    #[should_panic(expected = "report mode should be one of")]
    fn invalid_report() {
        let opt = r#"
            [map]
            name = "nullmap"

            [global]
            klen = 8
            vlen = 16
            kmin = 0
            kmax = 1000

            [[benchmark]]
            timeout = 1.0
            set_perc = 100
            get_perc = 0
            del_perc = 0
            scan_perc = 0
            dist = "incrementp"
            report = "alll"
        "#;

        let (_, _) = init(opt);
    }

    #[test]
    #[should_panic(expected = "cannot be provided at the same time")]
    fn invalid_length() {
        let opt = r#"
            [map]
            name = "nullmap"

            [global]
            klen = 8
            vlen = 16
            kmin = 0
            kmax = 1000

            [[benchmark]]
            timeout = 1.0
            ops = 1000
            set_perc = 100
            get_perc = 0
            del_perc = 0
            scan_perc = 0
            dist = "incrementp"
        "#;

        let (_, _) = init(opt);
    }

    #[test]
    #[should_panic(expected = "latency must also be true")]
    fn invalid_latency() {
        let opt = r#"
            [map]
            name = "nullmap"

            [global]
            klen = 8
            vlen = 16
            kmin = 0
            kmax = 1000

            [[benchmark]]
            timeout = 1.0
            cdf = true
            set_perc = 100
            get_perc = 0
            del_perc = 0
            scan_perc = 0
            dist = "incrementp"
        "#;

        let (_, _) = init(opt);
    }

    const EXAMPLE_BENCH: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/presets/benchmarks/example.toml"
    ));

    const EXAMPLE_SCAN_BENCH: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/presets/benchmarks/example_scan.toml"
    ));

    fn example(map_opt: &str) {
        let _ = env_logger::try_init();
        let opt = map_opt.to_string() + "\n" + EXAMPLE_BENCH;
        let (map, phases) = init(&opt);
        map.bench(&phases);
    }

    fn example_scan(map_opt: &str) {
        let _ = env_logger::try_init();
        let opt = map_opt.to_string() + "\n" + EXAMPLE_SCAN_BENCH;
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
    fn example_scan_null() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/null.toml"
        ));
        example_scan(OPT);
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
    fn example_scan_null_async() {
        const OPT: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/presets/stores/null_async.toml"
        ));
        example_scan(OPT);
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

    #[test]
    #[cfg(feature = "rocksdb")]
    fn example_rocksdb() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let opt = format!(
            r#"
            [map]
            name = "rocksdb"
            path = "{}"
            "#,
            tmp_dir.path().to_str().unwrap().to_string()
        );
        example(&opt);
    }

    #[test]
    #[cfg(feature = "rocksdb")]
    fn example_scan_rocksdb() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let opt = format!(
            r#"
            [map]
            name = "rocksdb"
            path = "{}"
            "#,
            tmp_dir.path().to_str().unwrap().to_string()
        );
        example_scan(&opt);
    }
}

// }}} tests

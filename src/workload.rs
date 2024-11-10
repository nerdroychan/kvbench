//! Workload generator.

use crate::Operation;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand::prelude::SliceRandom;
use rand::Rng;
use serde::Deserialize;
use zipf::ZipfDistribution;

/// This is for internal use in the workload mod. It is essentially Operation without
/// generated keys, values, or other parameters. They are generated based on a Mix defined below.
#[derive(Clone)]
enum OperationType {
    Set,
    Get,
    Delete,
    Scan,
}

/// Mix defines the percentages of operations, it consists of multiple supported operations
/// and the total of each operation should be 100.
/// As of now, it supports two access types, insert and read.
#[derive(Debug)]
struct Mix {
    dist: WeightedIndex<u8>,
}

impl Mix {
    fn new(set: u8, get: u8, delete: u8, scan: u8) -> Self {
        let dist = WeightedIndex::new(&[set, get, delete, scan]).unwrap();
        Self { dist }
    }

    fn next(&self, rng: &mut impl Rng) -> OperationType {
        let ops = [
            OperationType::Set,
            OperationType::Get,
            OperationType::Delete,
            OperationType::Scan,
        ];
        ops[self.dist.sample(rng)].clone()
    }
}

/// The distribution of keys, more distributions might be added.
#[derive(Debug)]
enum KeyDistribution {
    Increment,
    Shuffle(Vec<usize>),
    Uniform(Uniform<usize>),
    Zipfian(ZipfDistribution, usize),
    ZipfianLatest(ZipfDistribution, usize, usize),
}

/// Key generator that takes care of synthetic keys based on a distribution. Currently it only
/// generates fixed-sized keys based on the parameters of length and key space size.
#[derive(Debug)]
struct KeyGenerator {
    len: usize,
    min: usize,
    max: usize,
    keyspace: usize,
    serial: usize,
    dist: KeyDistribution,
}

/// Since we use `usize` for the numeric keys generated, the maximum key space size is limited by
/// the platform. If the target platform is 32-bit, all possible keys would have already filled the
/// memory, unless it is supporting a large persistent store, which is unlikely the case.
const KEY_BYTES: usize = std::mem::size_of::<usize>();

impl KeyGenerator {
    fn new(len: usize, min: usize, max: usize, dist: KeyDistribution) -> Self {
        let keyspace = max - min;
        let serial = 0;
        Self {
            len,
            min,
            max,
            keyspace,
            serial,
            dist,
        }
    }

    fn new_increment(len: usize, min: usize, max: usize) -> Self {
        let dist = KeyDistribution::Increment;
        Self::new(len, min, max, dist)
    }

    fn new_shuffle(len: usize, min: usize, max: usize) -> Self {
        let mut shuffle = (0..(max - min)).collect::<Vec<usize>>();
        shuffle.shuffle(&mut rand::thread_rng());
        let dist = KeyDistribution::Shuffle(shuffle);
        Self::new(len, min, max, dist)
    }

    fn new_uniform(len: usize, min: usize, max: usize) -> Self {
        let dist = KeyDistribution::Uniform(Uniform::new(0, max - min));
        Self::new(len, min, max, dist)
    }

    fn new_zipfian(len: usize, min: usize, max: usize, theta: f64, hotspot: f64) -> Self {
        let hotspot = (hotspot * (max - min - 1) as f64) as usize; // approx location for discrete keys
        let dist =
            KeyDistribution::Zipfian(ZipfDistribution::new(max - min, theta).unwrap(), hotspot);
        Self::new(len, min, max, dist)
    }

    fn new_zipfian_latest(len: usize, min: usize, max: usize, theta: f64, hotspot: f64) -> Self {
        let hotspot = (hotspot * (max - min - 1) as f64) as usize; // approx location for discrete keys
        let dist = KeyDistribution::ZipfianLatest(
            ZipfDistribution::new(max - min, theta).unwrap(),
            hotspot,
            0,
        );
        Self::new(len, min, max, dist)
    }

    fn next(&mut self, rng: &mut impl Rng) -> Box<[u8]> {
        let k = match self.dist {
            KeyDistribution::Increment => self.serial % self.keyspace,
            KeyDistribution::Shuffle(ref shuffle) => shuffle[self.serial % self.keyspace],
            KeyDistribution::Uniform(dist) => dist.sample(rng),
            KeyDistribution::Zipfian(dist, hotspot) => {
                // zipf starts at 1
                (dist.sample(rng) - 1 + hotspot) % self.keyspace
            }
            KeyDistribution::ZipfianLatest(dist, hotspot, ref mut latest) => {
                // just like zipfian, but always store the latest key
                let sample = dist.sample(rng) - 1;
                *latest = sample;
                (sample + hotspot) % self.keyspace
            }
        } + self.min;
        self.serial += 1;
        assert!(k < self.max);
        // fill 0s in the key to construct a key with length self.len
        let bytes = k.to_be_bytes();
        // key will hold the final key which is a Box<[u8]> and here we just do the allocation
        let mut key: Box<[u8]> = (0..self.len).map(|_| 0u8).collect();
        let len = self.len.min(KEY_BYTES);
        // copy from the big end to the beginning of the key slice
        key[0..len].copy_from_slice(&bytes[(KEY_BYTES - len)..KEY_BYTES]);
        key
    }
}

/// A set of workload parameters that can be deserialized from a TOML string.
///
/// **Note 1**: If an option not explicitly marked optional and it is not specified by both the file
/// and the global option, its default value will be applied. If it has no default value, an error
/// will be raised. The precedence of a value is: file > global (after env overridden) > default.
///
/// **Note 2**: the sum of all `*_perc` options must be equal to 100.
#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct WorkloadOpt {
    /// Percentage of `SET` operations.
    ///
    /// Must be a non-negative integer if given.
    ///
    /// Default: 0.
    pub set_perc: Option<u8>,

    /// Percentage of `GET` operations.
    ///
    /// Must be a non-negative integer if given.
    ///
    /// Default: 0.
    pub get_perc: Option<u8>,

    /// Percentage of `DELETE` operations.
    ///
    /// Must be a non-negative integer if given.
    ///
    /// Default: 0.
    pub del_perc: Option<u8>,

    /// Percentage of `SCAN` operations.
    ///
    /// Must be a non-negative integer if given.
    ///
    /// Default: 0.
    pub scan_perc: Option<u8>,

    /// The number of iterations per `SCAN`.
    ///
    /// Must be a positive integer if provided.
    ///
    /// Default: 10.
    pub scan_n: Option<usize>,

    /// Key length in bytes.
    ///
    /// Must be a positive integer.
    pub klen: Option<usize>,

    /// Value length in bytes.
    ///
    /// Must be a positive integer.
    pub vlen: Option<usize>,

    /// Minimum key.
    ///
    /// Must be a non-negative integer.
    pub kmin: Option<usize>,

    /// Maximum key.
    ///
    /// Must be greater than `kmin`.
    pub kmax: Option<usize>,

    /// Key distribution.
    ///
    /// - "increment": sequentially incrementing from `kmin` to `kmax`.
    /// - "incrementp": partitioned `increment`, where each thread takes a range of keys. For
    /// example, if there are two threads with `kmin` of 0 and `kmax` of 10, one thread will get an
    /// "increment" distribution from 0 to 5, and another one will get 6 to 10.
    /// - "shuffle": shuffled sequence from `kmin` to `kmax`. One key appears exactly once during
    /// an iteration of the whole key space. This is useful to randomly prefill keys.
    /// - "shufflep": partitioned `shuffle`, similar to "incrementp" but the keys are shuffled for
    /// each range/thread.
    /// - "uniform": uniformly random keys from `kmin` to `kmax`.
    /// - "zipfian": random keys from `kmin` to `kmax` following Zipfian distribution.
    /// - "latest": just like Zipfian but the hotspot is the latest key written to the store.
    pub dist: String,

    /// The theta parameter for Zipfian distribution.
    ///
    /// Default: 1.0.
    pub zipf_theta: Option<f64>,

    /// The hotspot location for Zipfian distribution.
    ///
    /// 0.0 means the first key. 0.5 means approximately the middle in the key space.
    ///
    /// Default: 0.0.
    pub zipf_hotspot: Option<f64>,
}

/// The minimal unit of workload context with its access pattern (mix and key generator).
///
/// The values generated internally are fixed-sized only for now, similar to the keys. To
/// pressurize the memory allocator, it might be a good idea to randomly adding a byte or two at
/// each generated values.
#[derive(Debug)]
pub(crate) struct Workload {
    /// Percentage of different operations
    mix: Mix,
    /// Key generator based on distribution
    kgen: KeyGenerator,
    /// Scan length
    scan_n: usize,
    /// Value length for operations that need a value
    vlen: usize,
    /// How many operations have been access so far
    count: u64,
}

impl Workload {
    pub fn new(opt: &WorkloadOpt, thread_info: Option<(usize, usize)>) -> Self {
        // input sanity checks
        let set_perc = opt.set_perc.unwrap_or(0);
        let get_perc = opt.get_perc.unwrap_or(0);
        let del_perc = opt.del_perc.unwrap_or(0);
        let scan_perc = opt.scan_perc.unwrap_or(0);
        assert_eq!(
            set_perc + get_perc + del_perc + scan_perc,
            100,
            "sum of ops in a mix should be 100"
        );
        let scan_n = opt.scan_n.unwrap_or(10);
        let klen = opt.klen.expect("klen should be specified");
        let vlen = opt.vlen.expect("vlen should be specified");
        let kmin = opt.kmin.expect("kmin should be specified");
        let kmax = opt.kmax.expect("kmax should be specified");
        assert!(scan_n > 0, "scan size should be positive");
        assert!(klen > 0, "klen should be positive");
        assert!(kmax > kmin, "kmax should be greater than kmin");

        let split_key_space = || {
            let (thread_id, nr_threads) = thread_info.expect("parallel keygen expects thread_info");
            assert!(thread_id < nr_threads);
            let nr_keys_per = (kmax - kmin) / nr_threads;
            let kminp = kmin + thread_id * nr_keys_per;
            let kmaxp = if thread_id == nr_threads - 1 {
                kmax
            } else {
                kminp + nr_keys_per
            };
            (kminp, kmaxp)
        };

        let mix = Mix::new(set_perc, get_perc, del_perc, scan_perc);
        let kgen = match opt.dist.as_str() {
            "increment" => KeyGenerator::new_increment(klen, kmin, kmax),
            "incrementp" => {
                let (kminp, kmaxp) = split_key_space();
                KeyGenerator::new_increment(klen, kminp, kmaxp)
            }
            "shuffle" => KeyGenerator::new_shuffle(klen, kmin, kmax),
            "shufflep" => {
                let (kminp, kmaxp) = split_key_space();
                KeyGenerator::new_shuffle(klen, kminp, kmaxp)
            }
            "uniform" => KeyGenerator::new_uniform(klen, kmin, kmax),
            "zipfian" => {
                let theta = opt.zipf_theta.unwrap_or(1.0f64);
                let hotspot = opt.zipf_hotspot.unwrap_or(0.0f64);
                KeyGenerator::new_zipfian(klen, kmin, kmax, theta, hotspot)
            }
            "latest" => {
                let theta = opt.zipf_theta.unwrap_or(1.0f64);
                let hotspot = opt.zipf_hotspot.unwrap_or(0.0f64);
                KeyGenerator::new_zipfian_latest(klen, kmin, kmax, theta, hotspot)
            }
            _ => {
                panic!("invalid key distribution: {}", opt.dist);
            }
        };
        Self {
            mix,
            kgen,
            scan_n,
            vlen,
            count: 0,
        }
    }

    pub fn next(&mut self, rng: &mut impl Rng) -> Operation {
        self.count += 1;
        let key = self.kgen.next(rng);
        match self.mix.next(rng) {
            OperationType::Set => {
                // special case for "latest" distribution: if the generated request is a `SET`,
                // update the hotspot to the latest generated key.
                if let KeyDistribution::ZipfianLatest(_, ref mut hotspot, latest) = self.kgen.dist {
                    *hotspot = latest;
                }
                let value = vec![0u8; self.vlen].into_boxed_slice();
                Operation::Set { key, value }
            }
            OperationType::Get => Operation::Get { key },
            OperationType::Delete => Operation::Delete { key },
            OperationType::Scan => Operation::Scan {
                key,
                n: self.scan_n,
            },
        }
    }

    pub fn reset(&mut self) {
        self.kgen.serial = 0;
    }

    pub fn is_exhausted(&self) -> bool {
        self.kgen.serial == self.kgen.keyspace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashbrown::{HashMap, HashSet};
    use quanta::Instant;

    #[test]
    fn mix_one_type_only() {
        let mut rng = rand::thread_rng();
        let mix = Mix::new(100, 0, 0, 0);
        for _ in 0..100 {
            assert!(matches!(mix.next(&mut rng), OperationType::Set));
        }
        let mix = Mix::new(0, 100, 0, 0);
        for _ in 0..100 {
            assert!(matches!(mix.next(&mut rng), OperationType::Get));
        }
        let mix = Mix::new(0, 0, 100, 0);
        for _ in 0..100 {
            assert!(matches!(mix.next(&mut rng), OperationType::Delete));
        }
        let mix = Mix::new(0, 0, 0, 100);
        for _ in 0..100 {
            assert!(matches!(mix.next(&mut rng), OperationType::Scan));
        }
    }

    #[test]
    fn mix_small_write() {
        let mut rng = rand::thread_rng();
        let mix = Mix::new(5, 95, 0, 0);
        let mut set = 0;
        #[allow(unused)]
        let mut get = 0;
        for _ in 0..1000000 {
            match mix.next(&mut rng) {
                OperationType::Set => set += 1,
                OperationType::Get => get += 1,
                OperationType::Delete => unreachable!(),
                OperationType::Scan => unreachable!(),
            };
        }
        assert!(set < 65000 && set > 35000);
    }

    #[test]
    fn keygen_increment() {
        // this also checks unaligned key
        let mut rng = rand::thread_rng();
        for len in [3, 8, 16] {
            let mut kgen = KeyGenerator::new_increment(len, 0, 3);
            let mut k: Box<[u8]> = (0..len).map(|_| 0u8).collect();
            for _ in 0..10 {
                k[len.min(8) - 1] = 0u8;
                assert_eq!(kgen.next(&mut rng), k);
                k[len.min(8) - 1] = 1u8;
                assert_eq!(kgen.next(&mut rng), k);
                k[len.min(8) - 1] = 2u8;
                assert_eq!(kgen.next(&mut rng), k);
            }
        }
    }

    #[test]
    fn keygen_shuffle() {
        let start = 117;
        let end = 135423;
        let mut rng = rand::thread_rng();
        let mut kgen = KeyGenerator::new_shuffle(8, start, end);
        let mut dist: HashSet<Box<[u8]>> = HashSet::new();
        for _ in start..end {
            let key = kgen.next(&mut rng);
            // the key is newly added, not repeated
            assert!(dist.insert(key.clone()));
        }
        // each key exists exactly once
        assert_eq!(dist.len(), end - start);
        let min = dist.iter().min().clone().unwrap();
        let min_bytes = Box::from(start.to_be_bytes());
        assert_eq!(*min, min_bytes);
        let max = dist.iter().max().clone().unwrap();
        let max_bytes = Box::from((end - 1).to_be_bytes());
        assert_eq!(*max, max_bytes);
    }

    #[test]
    fn keygen_uniform() {
        let mut rng = rand::thread_rng();
        let mut dist: HashMap<Box<[u8]>, u64> = HashMap::new();
        let mut kgen = KeyGenerator::new_uniform(8, 0, 100);
        // 100 keys, 1m gens so ~10k occurance ea. Bound to 9k to 11k
        // buy a lottry if this fails
        for _ in 0..1000000 {
            let k = kgen.next(&mut rng);
            dist.entry(k).and_modify(|c| *c += 1).or_insert(0);
        }
        for c in dist.values() {
            assert!(*c < 11000 && *c > 9000);
        }
    }

    #[test]
    fn keygen_zipfian() {
        let mut rng = rand::thread_rng();
        let mut dist: HashMap<Box<[u8]>, u64> = HashMap::new();
        let mut kgen = KeyGenerator::new_zipfian(8, 0, 10, 1.0, 0.0);
        for _ in 0..1000000 {
            let k = kgen.next(&mut rng);
            dist.entry(k).and_modify(|c| *c += 1).or_insert(0);
        }
        let mut freq: Vec<u64> = dist.values().map(|c| *c).collect();
        freq.sort_by_key(|c| std::cmp::Reverse(*c));
        // just some really nonsense checks
        let p1 = freq[0] as f64 / freq[1] as f64;
        assert!(p1 > 1.9 && p1 < 2.0, "zipf p1: {}", p1);
        let p2 = freq[1] as f64 / freq[2] as f64;
        assert!(p2 > 1.45 && p2 < 1.55, "zipf p2: {}", p2);
    }

    #[test]
    fn keygen_zipfian_hotspot() {
        let mut rng = rand::thread_rng();

        // hotspot is the middle key
        let mut dist: HashMap<Box<[u8]>, u64> = HashMap::new();
        let mut kgen = KeyGenerator::new_zipfian(8, 0, 9, 1.0, 0.5);
        for _ in 0..1000000 {
            let k = kgen.next(&mut rng);
            dist.entry(k).and_modify(|c| *c += 1).or_insert(0);
        }
        let mut freq: Vec<(Box<[u8]>, u64)> = dist.iter().map(|e| (e.0.clone(), *e.1)).collect();
        freq.sort_by_key(|c| c.0.clone());
        let p1 = freq[4].1 as f64 / freq[5].1 as f64;
        assert!(p1 > 1.9 && p1 < 2.0, "zipf p1: {}", p1);
        let p2 = freq[5].1 as f64 / freq[6].1 as f64;
        assert!(p2 > 1.45 && p2 < 1.55, "zipf p2: {}", p2);

        // hotspot is the last key
        let mut dist: HashMap<Box<[u8]>, u64> = HashMap::new();
        let mut kgen = KeyGenerator::new_zipfian(8, 0, 9, 1.0, 1.0);
        for _ in 0..1000000 {
            let k = kgen.next(&mut rng);
            dist.entry(k).and_modify(|c| *c += 1).or_insert(0);
        }
        let mut freq: Vec<(Box<[u8]>, u64)> = dist.iter().map(|e| (e.0.clone(), *e.1)).collect();
        freq.sort_by_key(|c| c.0.clone());
        let p1 = freq[8].1 as f64 / freq[0].1 as f64;
        assert!(p1 > 1.9 && p1 < 2.0, "zipf p1: {}", p1);
        let p2 = freq[0].1 as f64 / freq[1].1 as f64;
        assert!(p2 > 1.45 && p2 < 1.55, "zipf p2: {}", p2);
    }

    #[test]
    fn keygen_speed() {
        const N: usize = 1_000_0000;
        let mut rng = rand::thread_rng();
        let mut kgen = KeyGenerator::new_zipfian(8, 0, 1000, 1.0, 0.0);
        let t = Instant::now();
        for _ in 0..N {
            let _ = kgen.next(&mut rng);
        }
        println!("zipfian time: {} ms", t.elapsed().as_millis());
        let mut kgen = KeyGenerator::new_uniform(8, 0, 1000);
        let t = Instant::now();
        for _ in 0..N {
            let _ = kgen.next(&mut rng);
        }
        println!("uniform time: {} ms", t.elapsed().as_millis());
        let mut kgen = KeyGenerator::new_increment(8, 0, 1000);
        let t = Instant::now();
        for _ in 0..N {
            let _ = kgen.next(&mut rng);
        }
        println!("increment time: {} ms", t.elapsed().as_millis());
        let mut kgen = KeyGenerator::new_shuffle(8, 0, 1000);
        let t = Instant::now();
        for _ in 0..N {
            let _ = kgen.next(&mut rng);
        }
        println!("shuffle time: {} ms", t.elapsed().as_millis());
    }

    #[test]
    fn workload_toml_correct() {
        let s = r#"set_perc = 70
                   get_perc = 20
                   del_perc = 5
                   scan_perc = 5
                   scan_n = 100
                   klen = 4
                   vlen = 6
                   dist = "uniform"
                   kmin = 0
                   kmax = 12345"#;
        let opt: WorkloadOpt = toml::from_str(s).unwrap();
        let w = Workload::new(&opt, None);
        assert_eq!(w.scan_n, 100);
        assert_eq!(w.kgen.len, 4);
        assert_eq!(w.vlen, 6);
        assert_eq!(w.kgen.min, 0);
        assert_eq!(w.kgen.max, 12345);
        assert!(matches!(w.kgen.dist, KeyDistribution::Uniform(_)));

        let s = r#"set_perc = 70
                   get_perc = 20
                   del_perc = 5
                   scan_perc = 5
                   scan_n = 20
                   klen = 40
                   vlen = 60
                   dist = "zipfian"
                   kmin = 0
                   kmax = 123450
                   zipf_theta = 1.0
                   zipf_hotspot = 1.0"#;
        let opt: WorkloadOpt = toml::from_str(s).unwrap();
        let w = Workload::new(&opt, None);
        assert_eq!(w.scan_n, 20);
        assert_eq!(w.kgen.len, 40);
        assert_eq!(w.vlen, 60);
        assert_eq!(w.kgen.min, 0);
        assert_eq!(w.kgen.max, 123450);
        assert!(matches!(w.kgen.dist, KeyDistribution::Zipfian(_, 123449)));

        let s = r#"set_perc = 60
                   get_perc = 25
                   del_perc = 10
                   scan_perc = 5
                   scan_n = 30
                   klen = 14
                   vlen = 16
                   dist = "shuffle"
                   kmin = 10000
                   kmax = 20000"#;
        let opt: WorkloadOpt = toml::from_str(s).unwrap();
        let w = Workload::new(&opt, None);
        assert_eq!(w.scan_n, 30);
        assert_eq!(w.kgen.len, 14);
        assert_eq!(w.vlen, 16);
        assert_eq!(w.kgen.min, 10000);
        assert_eq!(w.kgen.max, 20000);
        assert!(matches!(w.kgen.dist, KeyDistribution::Shuffle(_)));
    }

    #[test]
    #[should_panic(expected = "should be positive")]
    fn workload_toml_invalid_wrong_size() {
        let s = r#"set_perc = 60
                   get_perc = 40
                   del_perc = 0
                   scan_perc = 0
                   scan_n = 10
                   klen = 0
                   vlen = 6
                   dist = "uniform"
                   kmin = 0
                   kmax = 12345"#;
        let opt: WorkloadOpt = toml::from_str(s).unwrap();
        let _ = Workload::new(&opt, None);
    }

    #[test]
    #[should_panic(expected = "should be positive")]
    fn workload_toml_invalid_wrong_scan() {
        let s = r#"set_perc = 60
                   get_perc = 40
                   del_perc = 0
                   scan_perc = 0
                   scan_n = 0
                   klen = 2
                   vlen = 6
                   dist = "uniform"
                   kmin = 0
                   kmax = 12345"#;
        let opt: WorkloadOpt = toml::from_str(s).unwrap();
        let _ = Workload::new(&opt, None);
    }

    #[test]
    #[should_panic(expected = "should be specified")]
    fn workload_toml_invalid_missing_fields() {
        let s = r#"set_perc = 60
                   get_perc = 40
                   del_perc = 0
                   scan_perc = 0
                   scan_n = 10
                   dist = "uniform"
                   kmin = 0
                   kmax = 12345"#;
        let opt: WorkloadOpt = toml::from_str(s).unwrap();
        let _ = Workload::new(&opt, None);
    }

    #[test]
    #[should_panic(expected = "should be greater")]
    fn workload_toml_invalid_wrong_keyspace() {
        let s = r#"set_perc = 60
                   get_perc = 40
                   del_perc = 0
                   scan_perc = 0
                   scan_n = 10
                   klen = 4
                   vlen = 6
                   dist = "uniform"
                   kmin = 5
                   kmax = 1"#;
        let opt: WorkloadOpt = toml::from_str(s).unwrap();
        let _ = Workload::new(&opt, None);
    }

    #[test]
    #[should_panic(expected = "should be 100")]
    fn workload_toml_invalid_wrong_mix() {
        let s = r#"set_perc = 70
                   get_perc = 40
                   del_perc = 0
                   scan_perc = 0
                   scan_n = 10
                   klen = 4
                   vlen = 6
                   dist = "uniform"
                   kmin = 0
                   kmax = 12345"#;
        let opt: WorkloadOpt = toml::from_str(s).unwrap();
        let _ = Workload::new(&opt, None);
    }

    #[test]
    #[should_panic(expected = "should be 100")]
    fn workload_toml_invalid_wrong_mix_missing() {
        let s = r#"klen = 4
                   vlen = 6
                   dist = "uniform"
                   kmin = 0
                   kmax = 12345"#;
        let opt: WorkloadOpt = toml::from_str(s).unwrap();
        let _ = Workload::new(&opt, None);
    }

    #[test]
    #[should_panic(expected = "invalid key distribution")]
    fn workload_toml_invalid_key_distribution() {
        let s = r#"set_perc = 70
                   get_perc = 30
                   del_perc = 0
                   scan_perc = 0
                   scan_n = 10
                   klen = 4
                   vlen = 6
                   dist = "uniorm"
                   kmin = 0
                   kmax = 12345"#;
        let opt: WorkloadOpt = toml::from_str(s).unwrap();
        let _ = Workload::new(&opt, None);
    }

    #[test]
    fn workload_toml_defaults_are_applied() {
        let s = r#"set_perc = 70
                   get_perc = 30
                   klen = 4
                   vlen = 6
                   dist = "zipfian"
                   kmin = 111
                   kmax = 12345"#;
        let opt: WorkloadOpt = toml::from_str(s).unwrap();
        let w = Workload::new(&opt, None);
        assert_eq!(w.scan_n, 10);
        assert!(matches!(w.kgen.dist, KeyDistribution::Zipfian(_, 0)));
    }

    #[test]
    fn workload_keygen_parallel() {
        let mut opt = WorkloadOpt {
            set_perc: Some(100),
            get_perc: None,
            del_perc: None,
            scan_perc: None,
            scan_n: Some(10),
            klen: Some(16),
            vlen: Some(100),
            dist: "incrementp".to_string(),
            kmin: Some(10000),
            kmax: Some(22347),
            zipf_theta: None,
            zipf_hotspot: None,
        };
        let test = |opt: &WorkloadOpt| {
            let workload = Workload::new(&opt, Some((0, 3)));
            assert_eq!(workload.kgen.min, 10000);
            assert_eq!(workload.kgen.max, 14115);

            let workload = Workload::new(&opt, Some((1, 3)));
            assert_eq!(workload.kgen.min, 14115);
            assert_eq!(workload.kgen.max, 18230);

            let workload = Workload::new(&opt, Some((2, 3)));
            assert_eq!(workload.kgen.min, 18230);
            assert_eq!(workload.kgen.max, 22347); // 2 more keys
        };
        // incrementp
        test(&opt);
        // shufflep
        opt.dist = "shufflep".to_string();
        test(&opt);
    }

    #[test]
    fn workload_keygen_parallel_fill() {
        let mut opt = WorkloadOpt {
            set_perc: Some(100),
            get_perc: None,
            del_perc: None,
            scan_perc: None,
            scan_n: Some(10),
            klen: Some(16),
            vlen: Some(100),
            dist: "incrementp".to_string(),
            kmin: Some(10000),
            kmax: Some(22347),
            zipf_theta: None,
            zipf_hotspot: None,
        };
        let test = |opt: &WorkloadOpt| {
            let mut rng = rand::thread_rng();
            let mut keys = HashSet::<Box<[u8]>>::new();
            let mut workloads: Vec<Workload> =
                (0..5).map(|t| Workload::new(&opt, Some((t, 5)))).collect();
            for w in workloads.iter_mut() {
                while !w.is_exhausted() {
                    let op = w.next(&mut rng);
                    let Operation::Set { key, .. } = op else {
                        panic!()
                    };
                    assert!(keys.insert(key));
                }
            }
            assert_eq!(keys.len(), 12347);
        };
        // incrementp
        test(&opt);
        // shufflep
        opt.dist = "shufflep".to_string();
        test(&opt);
    }

    #[test]
    fn workload_keygen_zipfian_latest() {
        let opt = WorkloadOpt {
            set_perc: Some(5),
            get_perc: Some(95),
            del_perc: None,
            scan_perc: None,
            scan_n: Some(10),
            klen: Some(16),
            vlen: Some(100),
            dist: "latest".to_string(),
            kmin: Some(10000),
            kmax: Some(22347),
            zipf_theta: None,
            zipf_hotspot: None,
        };

        let mut workload = Workload::new(&opt, None);
        let mut rng = rand::thread_rng();
        assert!(matches!(
            workload.kgen.dist,
            KeyDistribution::ZipfianLatest(_, 0, 0)
        ));
        let mut dist: HashSet<usize> = HashSet::new();
        let mut set_count = 0;
        for _ in 0..1000000 {
            let KeyDistribution::ZipfianLatest(_, this_hotspot, _) = workload.kgen.dist else {
                panic!();
            };
            dist.insert(this_hotspot);
            let op = workload.next(&mut rng);
            if let Operation::Set { key, value } = op {
                assert_eq!(key.len(), 16);
                assert_eq!(value.len(), 100);
                let KeyDistribution::ZipfianLatest(_, hotspot, latest) = workload.kgen.dist else {
                    panic!();
                };
                assert_eq!(hotspot, latest);
                set_count += 1;
            }
        }
        // approx. proportion of set should be ~5%.
        assert!(set_count < 60000);
        // and all the observed hotspot must be way less than that
        assert!(dist.len() < set_count);
    }

    #[test]
    fn workload_uniform_write_intensive() {
        let opt = WorkloadOpt {
            set_perc: Some(50),
            get_perc: Some(50),
            del_perc: None,
            scan_perc: None,
            scan_n: Some(10),
            klen: Some(16),
            vlen: Some(100),
            dist: "uniform".to_string(),
            kmin: Some(1000),
            kmax: Some(2000),
            zipf_theta: None,
            zipf_hotspot: None,
        };
        let mut workload = Workload::new(&opt, None);
        let mut set = 0;
        #[allow(unused)]
        let mut get = 0;
        let mut dist: HashMap<Box<[u8]>, u64> = HashMap::new();
        let mut rng = rand::thread_rng();
        for _ in 0..10000000 {
            let op = workload.next(&mut rng);
            match op {
                Operation::Set { key, value } => {
                    assert!(key.len() == 16);
                    assert!(value.len() == 100);
                    dist.entry(key).and_modify(|c| *c += 1).or_insert(0);
                    set += 1;
                }
                Operation::Get { key } => {
                    assert!(key.len() == 16);
                    dist.entry(key).and_modify(|c| *c += 1).or_insert(0);
                    get += 1;
                }
                Operation::Delete { .. } | Operation::Scan { .. } => {
                    unreachable!();
                }
            }
        }
        assert!(dist.keys().len() <= 1000);
        for c in dist.values() {
            assert!(*c < 12000 && *c > 8000);
        }
        assert!(set < 5500000 && set > 4500000);
    }

    #[test]
    fn workload_even_operations() {
        let opt = WorkloadOpt {
            set_perc: Some(25),
            get_perc: Some(25),
            del_perc: Some(25),
            scan_perc: Some(25),
            scan_n: None,
            klen: Some(16),
            vlen: Some(100),
            dist: "uniform".to_string(),
            kmin: Some(1000),
            kmax: Some(2000),
            zipf_theta: None,
            zipf_hotspot: None,
        };
        let mut workload = Workload::new(&opt, None);
        let mut set = 0;
        let mut get = 0;
        let mut del = 0;
        let mut scan = 0;
        let mut rng = rand::thread_rng();
        for _ in 0..10000000 {
            let op = workload.next(&mut rng);
            match op {
                Operation::Set { .. } => {
                    set += 1;
                }
                Operation::Get { .. } => {
                    get += 1;
                }
                Operation::Delete { .. } => {
                    del += 1;
                }
                Operation::Scan { .. } => {
                    scan += 1;
                }
            }
        }
        assert!(set > 2400000 && set < 2600000);
        assert!(get > 2400000 && get < 2600000);
        assert!(del > 2400000 && del < 2600000);
        assert!(scan > 2400000 && scan < 2600000);
    }
}

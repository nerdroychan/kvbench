//! Workload generator.

use crate::Operation;
use figment::providers::{Env, Format, Toml};
use figment::Figment;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
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
}

/// Mix defines the percentages of operations, it consists of multiple supported operations
/// and the total of each operation should be 100.
/// As of now, it supports two access types, insert and read.
#[derive(Debug)]
struct Mix {
    dist: WeightedIndex<u8>,
}

impl Mix {
    fn new(set: u8, get: u8, delete: u8) -> Self {
        let dist = WeightedIndex::new(&[set, get, delete]).unwrap();
        Self { dist }
    }

    fn next(&self, rng: &mut impl Rng) -> OperationType {
        let ops = [
            OperationType::Set,
            OperationType::Get,
            OperationType::Delete,
        ];
        ops[self.dist.sample(rng)].clone()
    }
}

/// The distribution of keys, more distributions might be added.
#[derive(Debug)]
enum KeyDistribution {
    Increment,
    Uniform(Uniform<usize>),
    Zipfian(ZipfDistribution, usize),
    // File,
}

/// Key generator that takes care of synthetic keys based on a distribution. Currently it only
/// generates fixed-sized keys based on the parameters of length and key space size.
#[derive(Debug)]
struct KeyGenerator {
    len: usize,
    min: usize,
    max: usize,
    keyspace: usize,
    dist: KeyDistribution,
    serial: usize,
}

impl KeyGenerator {
    fn new(len: usize, min: usize, max: usize, dist: KeyDistribution) -> Self {
        Self {
            len,
            min,
            max,
            keyspace: max - min,
            dist,
            serial: 0,
        }
    }

    fn new_increment(len: usize, min: usize, max: usize) -> Self {
        let dist = KeyDistribution::Increment;
        Self::new(len, min, max, dist)
    }

    fn new_uniform(len: usize, min: usize, max: usize) -> Self {
        let dist = KeyDistribution::Uniform(Uniform::new(0, max - min));
        Self::new(len, min, max, dist)
    }

    fn new_zipfian(len: usize, min: usize, max: usize, theta: f64, hotspot: f64) -> Self {
        let hotspot = (hotspot * (max - min) as f64) as usize; // approx location for discrete keys
        let dist =
            KeyDistribution::Zipfian(ZipfDistribution::new(max - min, theta).unwrap(), hotspot);
        Self::new(len, min, max, dist)
    }

    fn next(&mut self, rng: &mut impl Rng) -> Box<[u8]> {
        let key = match self.dist {
            KeyDistribution::Increment => self.serial % self.keyspace,
            KeyDistribution::Uniform(dist) => dist.sample(rng),
            KeyDistribution::Zipfian(dist, hotspot) => {
                (dist.sample(rng) + hotspot - 1) % self.keyspace
            } // zipf starts at 1
        } + self.min;
        self.serial += 1;
        assert!(key < self.max);
        // fill 0s in the key to construct a key with length self.len
        let bytes = key.to_be_bytes();
        // key will hold the final key which is a Box<[u8]> and here we just do the allocation
        let mut key: Box<[u8]> = (0..self.len).map(|_| 0u8).collect();
        let len = self.len.min(8);
        key[0..len].copy_from_slice(&bytes[8 - len..8]);
        key
    }
}

/// A set of workload parameters that can be deserialized from a TOML string.
///
/// This struct is used for interacting with workload configuration files and also create new
/// [`Workload`] instances.
#[derive(Deserialize, Clone, Debug)]
pub struct WorkloadOpt {
    /// Percentage of `SET` operations.
    pub set_perc: u8,

    /// Percentage of `GET` operations.
    pub get_perc: u8,

    /// Percentage of `DELETE` operations.
    pub del_perc: u8,

    /// Key length in bytes.
    pub klen: Option<usize>,

    /// Value length in bytes.
    pub vlen: Option<usize>,

    /// Minimum key.
    pub kmin: Option<usize>,

    /// Maximum key.
    pub kmax: Option<usize>,

    /// Key distribution.
    ///
    /// - "increment": sequentially incrementing from `kmin` to `kmax`.
    /// - "incrementp": partitioned `increment`, where each thread takes a range of keys.
    /// - "uniform": uniformly random keys from `kmin` to `kmax`.
    /// - "zipfian": random keys from `kmin` to `kmax` following Zipfian distribution.
    pub dist: String,

    /// The theta parameter for Zipfian distribution. (Optional, default 1.0)
    pub zipf_theta: Option<f64>,

    /// The hotspot location for Zipfian distribution. (Optional, default 0.0)
    ///
    /// 0.0 means the first key. 0.5 means approximately the middle in the key space.
    pub zipf_hotspot: Option<f64>,
}

/// The minimal unit of workload context with its access pattern (mix and key generator).
///
/// The values generated internally are fixed-sized only for now, similar to the keys. To
/// pressurize the
/// memory allocator, it might be a good idea to randomly adding a byte or two at each generated
/// values.
#[derive(Debug)]
pub struct Workload {
    /// Percentage of different operations
    mix: Mix,
    /// Key generator based on distribution
    kgen: KeyGenerator,
    /// Value length for operations that need a value
    vlen: usize,
    /// How many operations have been access so far
    count: u64,
}

impl Workload {
    pub fn new(opt: &WorkloadOpt, thread_info: Option<(usize, usize)>) -> Self {
        // input sanity checks
        assert_eq!(
            opt.set_perc + opt.get_perc + opt.del_perc,
            100,
            "sum of ops in a mix should be 100"
        );
        let klen = opt.klen.expect("klen should be specified");
        let vlen = opt.vlen.expect("vlen should be specified");
        let kmin = opt.kmin.expect("kmin should be specified");
        let kmax = opt.kmax.expect("kmax should be specified");
        assert!(klen > 0, "klen should be positive");
        assert!(kmax > kmin, "kmax should be greater than kmin");

        let mix = Mix::new(opt.set_perc, opt.get_perc, opt.del_perc);
        let kgen = match opt.dist.as_str() {
            "increment" => KeyGenerator::new_increment(klen, kmin, kmax),
            "incrementp" => {
                let (thread_id, nr_threads) = thread_info.expect("incrementp expects thread info");
                assert!(thread_id < nr_threads);
                let nr_keys_per = (kmax - kmin) / nr_threads;
                let kminp = kmin + thread_id * nr_keys_per;
                let kmaxp = if thread_id == nr_threads - 1 {
                    kmax
                } else {
                    kminp + nr_keys_per
                };
                KeyGenerator::new_increment(klen, kminp, kmaxp)
            }
            "uniform" => KeyGenerator::new_uniform(klen, kmin, kmax),
            "zipfian" => {
                let theta = opt.zipf_theta.unwrap_or(1.0f64);
                let hotspot = opt.zipf_hotspot.unwrap_or(0.0f64);
                KeyGenerator::new_zipfian(klen, kmin, kmax, theta, hotspot)
            }
            _ => {
                panic!("invalid key distribution: {}", opt.dist);
            }
        };
        Self {
            mix,
            kgen,
            vlen,
            count: 0,
        }
    }

    pub fn new_from_toml_str(text: &str, thread_info: Option<(usize, usize)>) -> Self {
        let opt: WorkloadOpt = Figment::new()
            .merge(Toml::string(text))
            .merge(Env::raw())
            .extract()
            .unwrap();
        Self::new(&opt, thread_info)
    }

    pub fn next(&mut self, rng: &mut impl Rng) -> Operation {
        self.count += 1;
        let key = self.kgen.next(rng);
        match self.mix.next(rng) {
            OperationType::Set => {
                let value = vec![0u8; self.vlen].into_boxed_slice();
                Operation::Set { key, value }
            }
            OperationType::Get => Operation::Get { key },
            OperationType::Delete => Operation::Delete { key },
        }
    }

    pub fn reset(&mut self) {
        self.kgen.serial = 0;
    }

    pub fn is_exhausted(&self) -> bool {
        self.kgen.serial == (self.kgen.max - self.kgen.min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashbrown::HashMap;
    use quanta::Instant;

    #[test]
    fn mix_one_type_only() {
        let mut rng = rand::thread_rng();
        let mix = Mix::new(100, 0, 0);
        for _ in 0..100 {
            assert!(matches!(mix.next(&mut rng), OperationType::Set));
        }
        let mix = Mix::new(0, 100, 0);
        for _ in 0..100 {
            assert!(matches!(mix.next(&mut rng), OperationType::Get));
        }
        let mix = Mix::new(0, 0, 100);
        for _ in 0..100 {
            assert!(matches!(mix.next(&mut rng), OperationType::Delete));
        }
    }

    #[test]
    fn mix_small_write() {
        let mut rng = rand::thread_rng();
        let mix = Mix::new(5, 95, 0);
        let mut set = 0;
        #[allow(unused)]
        let mut get = 0;
        for _ in 0..1000000 {
            match mix.next(&mut rng) {
                OperationType::Set => set += 1,
                OperationType::Get => get += 1,
                OperationType::Delete => unreachable!(),
            };
        }
        assert!(set < 65000 && set > 35000);
    }

    #[test]
    fn keygen_increment() {
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
        let mut dist: HashMap<Box<[u8]>, u64> = HashMap::new();
        let mut kgen = KeyGenerator::new_zipfian(8, 0, 10, 1.0, 0.5);
        for _ in 0..1000000 {
            let k = kgen.next(&mut rng);
            dist.entry(k).and_modify(|c| *c += 1).or_insert(0);
        }
        let mut freq: Vec<(Box<[u8]>, u64)> = dist.iter().map(|e| (e.0.clone(), *e.1)).collect();
        freq.sort_by_key(|c| c.0.clone());
        // should be 4 actually, but since the key space is discrete it is just approximate
        let p1 = freq[5].1 as f64 / freq[6].1 as f64;
        assert!(p1 > 1.9 && p1 < 2.0, "zipf p1: {}", p1);
        let p2 = freq[6].1 as f64 / freq[7].1 as f64;
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
    }

    #[test]
    fn workloadopt_toml_correct() {
        let s = r#"set_perc = 70
                   get_perc = 20
                   del_perc = 10
                   klen = 4
                   vlen = 6
                   dist = "zipfian"
                   kmin = 0
                   kmax = 12345"#;
        let w = Workload::new_from_toml_str(s, None);
        assert_eq!(w.kgen.min, 0);

        let s = r#"set_perc = 70
                   get_perc = 20
                   del_perc = 10
                   klen = 4
                   vlen = 6
                   dist = "uniform"
                   kmin = 0
                   kmax = 12345
                   zipf_theta = 1.0"#;
        let w = Workload::new_from_toml_str(s, None);
        assert_eq!(w.kgen.min, 0);
    }

    #[test]
    #[should_panic(expected = "should be positive")]
    fn workloadopt_toml_invalid_wrong_size() {
        let s = r#"set_perc = 60
                   get_perc = 40
                   del_perc = 0
                   klen = 0
                   vlen = 6
                   dist = "uniform"
                   kmin = 0
                   kmax = 12345"#;
        let _ = Workload::new_from_toml_str(s, None);
    }

    #[test]
    #[should_panic(expected = "should be specified")]
    fn workloadopt_toml_invalid_missing_fields() {
        let s = r#"set_perc = 60
                   get_perc = 40
                   del_perc = 0
                   dist = "uniform"
                   kmin = 0
                   kmax = 12345"#;
        let _ = Workload::new_from_toml_str(s, None);
    }

    #[test]
    #[should_panic(expected = "should be greater")]
    fn workloadopt_toml_invalid_wrong_keyspace() {
        let s = r#"set_perc = 60
                   get_perc = 40
                   del_perc = 0
                   klen = 4
                   vlen = 6
                   dist = "uniform"
                   kmin = 5
                   kmax = 1"#;
        let _ = Workload::new_from_toml_str(s, None);
    }

    #[test]
    #[should_panic(expected = "should be 100")]
    fn workloadopt_toml_invalid_wrong_mix() {
        let s = r#"set_perc = 70
                   get_perc = 40
                   del_perc = 0
                   klen = 4
                   vlen = 6
                   dist = "uniform"
                   kmin = 0
                   kmax = 12345"#;
        let _ = Workload::new_from_toml_str(s, None);
    }

    #[test]
    fn workload_increment_parallel() {
        let opt = WorkloadOpt {
            set_perc: 100,
            get_perc: 0,
            del_perc: 0,
            klen: Some(16),
            vlen: Some(100),
            dist: "incrementp".to_string(),
            kmin: Some(10000),
            kmax: Some(22347),
            zipf_theta: None,
            zipf_hotspot: None,
        };

        let workload = Workload::new(&opt, Some((0, 3)));
        assert_eq!(workload.kgen.min, 10000);
        assert_eq!(workload.kgen.max, 14115);

        let workload = Workload::new(&opt, Some((1, 3)));
        assert_eq!(workload.kgen.min, 14115);
        assert_eq!(workload.kgen.max, 18230);

        let workload = Workload::new(&opt, Some((2, 3)));
        assert_eq!(workload.kgen.min, 18230);
        assert_eq!(workload.kgen.max, 22347); // 2 more keys
    }

    #[test]
    fn workload_uniform_write_intensive() {
        let opt = WorkloadOpt {
            set_perc: 50,
            get_perc: 50,
            del_perc: 0,
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
                Operation::Delete { .. } => {
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
}

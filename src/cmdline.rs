use crate::stores::Registry;
use clap::ValueHint::FilePath;
use clap::{Args, Parser, Subcommand};
use log::debug;
use std::fs::read_to_string;
use std::sync::mpsc::channel;

#[derive(Args, Debug)]
struct BenchArgs {
    #[arg(short = 's')]
    #[arg(value_hint = FilePath)]
    #[arg(help = "Path to the key-value store's TOML config file")]
    store_config: String,

    #[arg(short = 'b')]
    #[arg(value_hint = FilePath)]
    #[arg(help = "Path to the benchmark's TOML config file")]
    benchmark_config: String,
}

#[derive(Args, Debug)]
struct ServerArgs {
    #[arg(short = 'a', default_value = "0.0.0.0")]
    #[arg(help = "Bind address")]
    host: String,

    #[arg(short = 'p', default_value = "9000")]
    #[arg(help = "Bind port")]
    port: String,

    #[arg(short = 's')]
    #[arg(value_hint = FilePath)]
    #[arg(help = "Path to the key-value store's TOML config file")]
    store_config: String,

    #[arg(short = 'n', default_value_t = 1)]
    #[arg(help = "Number of worker threads")]
    workers: usize,
}

#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(about = "Run a benchmark")]
    Bench(BenchArgs),
    #[command(about = "Start a key-value server")]
    Server(ServerArgs),
    #[command(about = "List all registered key-value stores")]
    List,
}

fn bench_cli(args: &BenchArgs) {
    let opt: String = {
        let s = args.store_config.clone();
        let b = args.benchmark_config.clone();
        read_to_string(s.as_str()).unwrap() + "\n" + &read_to_string(b.as_str()).unwrap()
    };

    let (map, phases) = crate::bench::init(&opt);
    map.bench(&phases);
}

fn server_cli(args: &ServerArgs) {
    let host = &args.host;
    let port = &args.port;
    let nr_workers = args.workers;

    let opt: String = read_to_string(args.store_config.as_str()).unwrap();
    let map = crate::server::init(&opt);

    let (stop_tx, stop_rx) = channel();
    let (grace_tx, grace_rx) = channel();

    ctrlc::set_handler(move || {
        assert!(stop_tx.send(()).is_ok());
        debug!("SIGINT received and stop message sent to server");
    })
    .expect("Error setting Ctrl-C handler for server");

    map.server(&host, &port, nr_workers, stop_rx, grace_tx);

    assert!(grace_rx.recv().is_ok());
    debug!("All server threads have been shut down gracefully, exit");
}

fn list_cli() {
    for r in inventory::iter::<Registry> {
        println!("Registered map: {}", r.name);
    }
}

/// The default command line interface.
///
/// This function is public and can be called in a different crate. For example, one can integrate
/// their own key-value stores by registering the constructor function. Then, adding this function
/// will produce a benchmark binary the has the same usage as the one in this crate.
///
/// ## Usage
///
/// To get the usage of the command line interface, users can run:
///
/// ```bash
/// kvbench -h
/// ```
///
/// The interface supports three modes, `bench`, `server` and `list`.
///
/// ### Benchmark Mode
///
/// Usage:
///
/// ```bash
/// kvbench bench -s <STORE_CONFIG> -b <BENCH_CONFIG>
/// ```
///
/// Where `STORE_CONFIG` and `BENCH_CONFIG` are the paths to the key-value store and benchmark
/// configuration files, respectively. For their format, you can refer to the documentations of
/// [`crate::stores`] and [`crate::bench`].
///
/// ### Server mode
///
/// Usage:
///
/// ```bash
/// kvbench server -s <STORE_CONFIG> -a <HOST> -p <PORT> -n <WORKERS>
/// ```
///
/// Where `STORE_CONFIG` is the path of the key-value store configuration file. Its format is
/// documented in [`crate::stores`].
///
/// The default `HOST` and `PORT` are `0.0.0.0` and `9000`. By default, the server will spawn one
/// worker thread only for incoming connections. You can adjust the number of worker threads by
/// specifying `-n`.
///
/// ### List mode
///
/// Usage:
/// ``` bash
/// kvbench list
/// ```
///
/// This command lists all registered key-value stores' names.

pub fn cmdline() {
    env_logger::init();
    let cli = Cli::parse();
    debug!("Starting kvbench with args: {:?}", cli);
    match cli.command {
        Commands::Bench(args) => bench_cli(&args),
        Commands::Server(args) => server_cli(&args),
        Commands::List => list_cli(),
    }
}

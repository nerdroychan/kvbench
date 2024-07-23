use crate::bench::*;
use clap::ValueHint::FilePath;
use clap::{Args, Parser, Subcommand};
use log::debug;
use std::fs::read_to_string;
use std::sync::mpsc::channel;

#[derive(Args, Debug)]
struct BenchArgs {
    #[arg(short = 's')]
    #[arg(value_hint = FilePath)]
    store_config: String,

    #[arg(short = 'b')]
    #[arg(value_hint = FilePath)]
    benchmark_config: String,
}

#[derive(Args, Debug)]
struct ServerArgs {
    #[arg(short = 'h', default_value = "0.0.0.0")]
    host: String,

    #[arg(short = 'p', default_value = "9000")]
    port: String,

    #[arg(short = 's')]
    #[arg(value_hint = FilePath)]
    store_config: String,

    #[arg(short = 'n', default_value_t = 1)]
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
    Server(ServerArgs),
    Bench(BenchArgs),
}

fn bench_cli(args: &BenchArgs) {
    let opt: String = {
        let s = args.store_config.clone();
        let b = args.benchmark_config.clone();
        read_to_string(s.as_str()).unwrap() + "\n" + &read_to_string(b.as_str()).unwrap()
    };

    let (map, phases) = init(&opt);
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

    match map {
        BenchKVMap::Regular(map) => {
            map.server(&host, &port, nr_workers, stop_rx, grace_tx);
        }
        BenchKVMap::Async(map) => {
            map.server(&host, &port, nr_workers, stop_rx, grace_tx);
        }
    }

    assert!(grace_rx.recv().is_ok());
    debug!("All server threads have been shut down gracefully, exit");
}

/// The default command line interface.
///
/// This function is public and can be called in a different crate. For example, one can integrate
/// their own key-value stores by registering the constructor function. Then, adding this function
/// will produce a benchmark binary the has the same usage as the one in this crate.
pub fn cmdline() {
    env_logger::init();
    let cli = Cli::parse();
    debug!("Starting kvbench with args: {:?}", cli);
    match cli.command {
        Commands::Bench(args) => bench_cli(&args),
        Commands::Server(args) => server_cli(&args),
    }
}

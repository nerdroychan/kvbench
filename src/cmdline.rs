use crate::bench::*;
use clap::ValueHint::FilePath;
use clap::{Args, Parser, Subcommand};
use log::debug;
use std::fs::read_to_string;
use std::sync::mpsc::channel;

#[derive(Args, Debug)]
struct BenchArgs {
    #[arg(short = 'f', group = "combined", conflicts_with = "separate")]
    #[arg(required_unless_present_any = ["map_file", "benchmark_file"])]
    #[arg(value_hint = FilePath)]
    file: Option<String>,

    #[arg(short = 'm', group = "separate", conflicts_with = "combined")]
    #[arg(requires = "benchmark_file")]
    #[arg(value_hint = FilePath)]
    map_file: Option<String>,

    #[arg(short = 'b', group = "separate", conflicts_with = "combined")]
    #[arg(requires = "map_file")]
    #[arg(value_hint = FilePath)]
    benchmark_file: Option<String>,
}

#[derive(Args, Debug)]
struct ServerArgs {
    #[arg(short = 'h', default_value = "0.0.0.0")]
    host: String,

    #[arg(short = 'p', default_value = "9000")]
    port: String,

    #[arg(short = 'm')]
    #[arg(value_hint = FilePath)]
    map_file: String,

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
    let opt: String = if let Some(f) = &args.file {
        read_to_string(f.as_str()).unwrap()
    } else {
        let m = args.map_file.clone().unwrap();
        let b = args.benchmark_file.clone().unwrap();
        read_to_string(m.as_str()).unwrap() + "\n" + &read_to_string(b.as_str()).unwrap()
    };

    let (map, phases) = init(&opt);
    map.bench(&phases);
}

fn server_cli(args: &ServerArgs) {
    let host = &args.host;
    let port = &args.port;
    let nr_workers = args.workers;

    let opt: String = read_to_string(args.map_file.as_str()).unwrap();
    let map = crate::server::init(&opt);

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

pub fn default() {
    env_logger::init();
    let cli = Cli::parse();
    debug!("Starting kvbench with args: {:?}", cli);
    match cli.command {
        Commands::Bench(args) => bench_cli(&args),
        Commands::Server(args) => server_cli(&args),
    }
}

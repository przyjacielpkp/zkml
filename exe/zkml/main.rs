use lib::*;

use clap::{Parser, Subcommand};
use std::{error::Error, path::PathBuf};

#[derive(Parser)]
struct Cli {
  #[command(subcommand)]
  command: Command,
}

#[derive(Subcommand)]
enum Command {
  /// ZKML prover
  Client {
    /// File with input to be classified
    #[arg(long, value_name = "PATH")]
    input: PathBuf,
    /// File with proving key
    #[arg(long, default_value = "pk.in")]
    proving_key: PathBuf,
    /// File with neural network weights
    #[arg(long, default_value = "weights.in")]
    weights: PathBuf,
    /// URL of verifier
    #[arg(long)]
    url: String,
    #[arg(short, long, default_value_t = 4545)]
    port: u16,
  },
  /// ZKML verifier
  Server {
    /// Port to run server on
    #[arg(short, long, default_value_t = 4545)]
    port: u16,
    /// File with neural network weights
    #[arg(long, default_value = "weights.in")]
    weights: PathBuf,
    /// File with verifying key
    #[arg(long, default_value = "vk.in")]
    verifying_key: PathBuf,
  },
  Setup {
    /// Input file with dataset
    #[arg(long)]
    dataset: PathBuf,
    /// File to save proving key in
    #[arg(long, default_value = "pk.in")]
    proving_key: PathBuf,
    /// File to save verifying key in
    #[arg(long, default_value = "vk.in")]
    verifying_key: PathBuf,
    /// File to save weights in
    #[arg(long, default_value = "weights.in")]
    weights: PathBuf,
  },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
  utils::init_logging()?;
  let args = Cli::parse();

  match args.command {
    Command::Client {
      input,
      proving_key,
      weights,
      url,
      port,
    } => {
      let true_url = format!("https://{}:{}/", url, port);
      let app = subcommands::Client::new(input, proving_key, weights, true_url);
      app.run().await;
    }
    Command::Server {
      port,
      weights,
      verifying_key,
    } => {
      let app = subcommands::Server::new(port, weights, verifying_key);
      app.run().await;
    }
    Command::Setup {
      dataset,
      proving_key,
      verifying_key,
      weights,
    } => {
      let app = subcommands::Setup::new(&dataset, &proving_key, &verifying_key, &weights);
      app.run();
    }
  }
  Ok(())
}

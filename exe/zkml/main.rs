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
  /// Honest ZKML prover
  Client {
    /// URL of verifier
    #[arg(long)]
    url: String,
    #[arg(short, long, default_value_t = 4545)]
    port: u16,
    /// File with input to be classified
    #[arg(long, default_value = "input.txt")]
    input: PathBuf,
    /// File with proving key
    #[arg(long, default_value = "pk.txt")]
    pk: PathBuf,
    /// File with model weights
    #[arg(long, default_value = "weights.txt")]
    weights: PathBuf,
  },
  /// ZKML verifier
  Server {
    #[arg(short, long, default_value_t = 4545)]
    port: u16,
    // File with verifying key
    #[arg(long, default_value = "vk.txt")]
    vk: PathBuf,
  },
  /// Train ML model
  Model {
    #[arg(short, long, default_value_t = 20)]
    epochs: usize,
    /// File with dataset
    #[arg(long, default_value = "data/rp.data")]
    dataset: PathBuf,
    /// Output file for proving key
    #[arg(long, default_value = "pk.txt")]
    pk: PathBuf,
    /// Output file for verifying key
    #[arg(long, default_value = "vk.txt")]
    vk: PathBuf,
    /// Output file for model weights
    #[arg(long, default_value = "weights.txt")]
    weights: PathBuf,
  },
  /// ZKML prover that sends doctored input to the verifier
  Doctor {
    /// URL of verifier
    #[arg(long)]
    url: String,
    #[arg(short, long, default_value_t = 4545)]
    port: u16,
    /// File with proving key
    #[arg(long, default_value = "pk.txt")]
    pk: PathBuf,
    /// File with model weights
    #[arg(long, default_value = "weights.txt")]
    weights: PathBuf,
  },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
  utils::init_logging()?;
  let args = Cli::parse();

  match args.command {
    Command::Client {
      url,
      port,
      input,
      pk,
      weights,
    } => {
      let true_url = format!("http://{}:{}/", url, port);
      let app = subcommands::Client::new(&input, &pk, &weights, true_url);
      app.run().await;
    }
    Command::Server { port, vk } => {
      let app = subcommands::Server::new(port, &vk);
      app.run().await;
    }
    Command::Model {
      dataset,
      vk,
      pk,
      weights,
      epochs,
    } => {
      let app = subcommands::Setup::new(&dataset, &pk, &vk, &weights, epochs);
      app.run();
    }
    Command::Doctor {
      url,
      port,
      pk,
      weights,
    } => {
      let true_url = format!("http://{}:{}/", url, port);
      let app = subcommands::Doctor::new(&pk, &weights, true_url);
      app.run().await;
    }
  }
  Ok(())
}

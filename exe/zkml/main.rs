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
    /// URL of verifier
    #[arg(long)]
    url: String,
    #[arg(short, long, default_value_t = 4545)]
    port: u16,
    /// File with input to be classified
    #[arg(long)]
    input: PathBuf,
    /// File with proving key
    #[arg(long)]
    pk: PathBuf,
    /// File with input to be classified
    #[arg(long)]
    model: PathBuf,
  },
  /// ZKML verifier
  Server {
    #[arg(short, long, default_value_t = 4545)]
    port: u16,
    // File with verifying key
    #[arg(long)]
    vk: PathBuf,
  },
  /// Train ML model
  Model {
    #[arg(short, long, default_value_t = 20)]
    epochs: usize,
    /// File with dataset
    #[arg(long)]
    dataset: PathBuf,
    /// Output file for proving key
    #[arg(long)]
    pk: PathBuf,
    /// Output file for verifying key
    #[arg(long)]
    vk: PathBuf,
    /// Output file for weights
    #[arg(long)]
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
      model,
    } => {
      let true_url = format!("http://{}:{}/", url, port);
      let app = subcommands::Client::new(&input, &pk, &model, true_url);
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
  }
  Ok(())
}

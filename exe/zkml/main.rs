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
    input_file: PathBuf,
    /// URL of verifier
    #[arg(long)]
    url: String,
    #[arg(short, long, default_value_t = 4545)]
    port: u16,
  },
  /// ZKML verifier
  Server {
    #[arg(short, long, default_value_t = 4545)]
    port: u16,
  },
  Model {
    #[arg(long)]
    dataset_path: PathBuf,
    #[arg(long)]
    prover_output_path: PathBuf,
    #[arg(long)]
    verifier_output_path: PathBuf,
    #[arg(long)]
    weights_output_path: PathBuf,
  },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
  utils::init_logging()?;
  let args = Cli::parse();

  match args.command {
    Command::Client {
      input_file,
      url,
      port,
    } => {
      let true_url = format!("https://{}:{}/", url, port);
      let app = subcommands::Client::new(input_file, true_url);
      app.run().await;
    }
    Command::Server { port } => {
      let app = subcommands::Server::new(port);
      app.run().await;
    }
    Command::Model {
      dataset_path,
      prover_output_path,
      verifier_output_path,
      weights_output_path,
    } => {
      let app = subcommands::Setup::new(
        &dataset_path,
        &prover_output_path,
        &verifier_output_path,
        &weights_output_path,
      );
      app.run();
    }
  }
  Ok(())
}

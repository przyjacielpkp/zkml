use lib::*;

use std::{error::Error, path::PathBuf};

use clap::{Parser, Subcommand};

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
  },
  /// ZKML verifier
  Server {},
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
  let args = Cli::parse();

  match args.command {
    Command::Client { input_file, url } => {
      let app = subcommands::Client::new(input_file, url);
      app.run();
    }
    Command::Server {} => {
      let app = subcommands::Server::new();
      app.run();
    }
  }
  Ok(())
}

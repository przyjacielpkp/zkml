use app_config::AppConfig;
#[cfg(not(debug_assertions))]
use human_panic::setup_panic;

#[cfg(debug_assertions)]
extern crate better_panic;

use tracing::subscriber::SetGlobalDefaultError;

use std::{fs::read_to_string, path::PathBuf};

use clap::{Parser, Subcommand};

mod app_config;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Optional name to operate on
    // name: Option<String>,

    /// Sets a custom config file
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    #[arg(short, long, default_value = "false")]
    some_option: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    // list commands here:

    /// does something
    Compile {

      // #[command(flatten)]
      // delegate: Struct,

      /// the argument file 
      #[arg(short, long, value_name = "FILE")]
      file: Option<PathBuf>,
      
      /// produce artifacts
      #[arg(long, value_name = "BOOL")]
      artifacts: Option<bool>,
    },
    /// does something else
    Whatever {},
    /// shows config
    Config {},
}

#[test]
fn verify_cli() {
    use clap::CommandFactory;
    Cli::command().debug_assert()
}

#[allow(dead_code)]
#[derive(Debug)]
enum Error {
  LoggerInitError(SetGlobalDefaultError),
  ConfigDeserializationError(serde_yaml::Error),
  ConfigFileOpenError(std::io::Error),
  CompileError(lib::Error),
  MissingArgument(String),
}

impl From<SetGlobalDefaultError> for Error {
  fn from(e: SetGlobalDefaultError) -> Self {
    Error::LoggerInitError(e)
  }
}

impl From<serde_yaml::Error> for Error {
  fn from(e: serde_yaml::Error) -> Self {
    Error::ConfigDeserializationError(e)
  }
}

impl From<lib::Error> for Error {
  fn from(e: lib::Error) -> Self {
    Error::CompileError(e)
  }
}

impl From<std::io::Error> for Error {
  fn from(e: std::io::Error) -> Self {
    Error::ConfigFileOpenError(e)
  } 
}

fn cli_match(command : Commands, config : AppConfig) -> Result<(), Error> {
  use Commands::*;
  match command {
    Whatever {  } => {
      // let whatever_config = lib::WhateverConfig {  };
      // etc
      tracing::info!("Doing whatever"); Ok(())
    },
    Compile { file: _file, artifacts } => {
      let cli_config = AppConfig {
        artifacts,
        ..AppConfig::default()
      };
      let config = config.merge(cli_config);
      let compile_config = {
        let artifacts = config.artifacts.ok_or(Error::MissingArgument("artifacts".to_string()))?;
        lib::CompileConfig {
          artifacts,
        }
      };
      lib::main(compile_config)?; Ok(())
    },
    Config {  } => {
      println!("{:?}", config);
      Ok(())
    },
  }
}

fn main() -> Result<(), Error> {
  utils::common_inits::init_logging()?;
  
  let Cli { config, some_option, command } = Cli::parse();

  // Merge configuration following from: cli options, config file, default (from resources/default_config.yaml)
  // in this order of priority.
  // Then match the cli command, merge additional settings provided with the command and run the command imported from lib.
  let default_config: AppConfig =
    serde_yaml::from_str(include_str!("resources/default_config.yaml"))?;
  let user_config = if let Some(config_path) = config {
    let contents = read_to_string(config_path.clone())?;
    serde_yaml::from_str(&contents)?
  } else {
    AppConfig::default()
  };
  let cli_config = AppConfig {
    some_option: Some(some_option),
    ..AppConfig::default()
  };
  let config = default_config.merge(user_config).merge(cli_config);
  cli_match(command, config)
} 

use ark_serialize::{CanonicalSerialize, Write};
use serde::Serialize;
use std::path::Path;

#[cfg(not(debug_assertions))]
use human_panic::setup_panic;
use tracing::subscriber::{self, SetGlobalDefaultError};

#[cfg(debug_assertions)]
extern crate better_panic;

use tracing_subscriber::{self, fmt};

// [NOTE] tracing
//
// In code use:
//
// use std::{error::Error, io};
// use tracing::{trace, debug, info, warn, error, Level};

// // the `#[tracing::instrument]` attribute creates and enters a span
// // every time the instrumented function is called. The span is named after
// // the function or method. Parameters passed to the function are recorded as fields.
// #[tracing::myfn]
// pub fn myfn\

pub fn install_logger() -> Result<(), SetGlobalDefaultError> {
  // let subscriber = tracing_subscriber::registry()
  //   .with(fmt::layer())
  //   .with(tracing_subscriber::filter::EnvFilter::from_default_env())
  //   .init();
  let subscriber = tracing_subscriber::fmt().compact();
  // .with
  // .with(EnvFilter::from_default_env());
  // let s = tracing_subscriber::registry().with(fmt::layer());

  // #[cfg(debug_assertions)]
  // let subscriber = subscriber.with_max_level(tracing::Level::DEBUG);

  let subscriber = subscriber.finish();
  return tracing::subscriber::set_global_default(subscriber);
}

pub fn init_logging() -> Result<(), SetGlobalDefaultError> {
  // Human Panic. Only enabled when *not* debugging.
  #[cfg(not(debug_assertions))]
  {
    setup_panic!();
  }

  // Better Panic. Only enabled *when* debugging.
  #[cfg(debug_assertions)]
  {
    better_panic::Settings::debug()
      .most_recent_first(false)
      .lineno_suffix(true)
      .verbosity(better_panic::Verbosity::Full)
      .install();
  }

  // Setup Logging
  install_logger()?;

  Ok(())
}

pub fn canonical_serialize_to_file<T: CanonicalSerialize>(path: &Path, obj: &T) {
  let mut buff = Vec::<u8>::new();
  obj.serialize(&mut buff);

  if let Err(e) = std::fs::write(path, buff) {
    panic!(
      "Error creating file {}: {}",
      path.to_str().unwrap(),
      e.to_string()
    );
  };
}

pub fn serialize_to_file<T: Serialize>(path: &Path, obj: &T) {
  let buff = serde_json::to_string(obj).unwrap();

  if let Err(e) = std::fs::write(path, buff) {
    panic!(
      "Error creating file {}: {}",
      path.to_str().unwrap(),
      e.to_string()
    );
  };
}

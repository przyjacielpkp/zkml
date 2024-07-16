#[cfg(not(debug_assertions))]
use human_panic::setup_panic;

#[cfg(debug_assertions)]
extern crate better_panic;

use tracing::subscriber::{DefaultGuard, SetGlobalDefaultError};
use tracing_subscriber::{self};

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

pub fn init_logging_tests() -> DefaultGuard {
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

  let subscriber = tracing_subscriber::fmt().compact();
  let subscriber = subscriber.finish();
  tracing::subscriber::set_default(subscriber)
}

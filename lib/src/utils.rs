
#[cfg(not(debug_assertions))]
use human_panic::setup_panic;
use tracing::subscriber::SetGlobalDefaultError;

#[cfg(debug_assertions)]
extern crate better_panic;

use tracing_subscriber;

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
    let subscriber = tracing_subscriber::fmt().finish();
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
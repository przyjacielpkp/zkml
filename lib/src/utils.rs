use std::path::Path;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
#[cfg(not(debug_assertions))]
use human_panic::setup_panic;

#[cfg(debug_assertions)]
extern crate better_panic;

use luminal::prelude::NodeIndex;
use serde::{de::DeserializeOwned, Serialize};
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
  tracing::subscriber::set_global_default(subscriber)
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

pub fn unpack_node_index(idx: &NodeIndex) -> u32 {
  // this is bad but unfortunatelly necessary
  let str = format!("{:?}", idx);

  if &str[0..10] == "NodeIndex(" && &str[str.len() - 1..] == ")" {
    let substr = &str[10..str.len() - 1];
    substr.parse::<u32>().expect("Failed to unpack node index")
  } else {
    panic!("Failed to unpack node index")
  }
}

pub fn canonical_serialize_to_file<T: CanonicalSerialize>(path: &Path, obj: &T) {
  let mut buff = Vec::<u8>::new();
  obj
    .serialize(&mut buff)
    .expect("Object serialization failed");

  serialize_to_file(path, &buff);
}

pub fn serialize_to_file<T: Serialize>(path: &Path, obj: &T) {
  let buff = serde_json::to_string(obj).unwrap();

  if let Err(e) = std::fs::write(path, buff) {
    panic!(
      "Error while writing to file {}: {}",
      path.to_str().unwrap(),
      e
    );
  };
}

pub fn canonical_deserialize_from_file<T: CanonicalDeserialize>(path: &Path) -> T {
  let buff: Vec<u8> = deserialize_from_file(path);
  T::deserialize(buff.as_slice()).expect("Object deserialization failed")
}

pub fn deserialize_from_file<T: DeserializeOwned>(path: &Path) -> T {
  match std::fs::read(path) {
    Ok(buff) => serde_json::from_slice(&buff).expect("Object deserialization failed"),
    Err(e) => panic!("Failed to read file {}: {}", path.to_str().unwrap(), e),
  }
}

pub fn serialize_model_to_file<T: Serialize>(path: &Path, obj: &T) {
  let buff = serde_json::to_string(obj).unwrap();

  if let Err(e) = std::fs::write(path, buff) {
    panic!(
      "Error while writing to file {}: {}",
      path.to_str().unwrap(),
      e
    );
  }
}

#[cfg(test)]
mod tests {
  use luminal::prelude::NodeIndex;

  use crate::utils::unpack_node_index;

  #[test]
  pub fn should_unpack_node_index() {
    let idx = NodeIndex::from(43u32);

    assert_eq!(43, unpack_node_index(&idx));
  }
}

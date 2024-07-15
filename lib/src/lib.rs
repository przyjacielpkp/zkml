use std::collections::HashMap;

use itertools::Itertools;
use luminal::prelude::NodeIndex;
use model::{get_weights, Model, TrainedGraph};
use scalar::{scalar, InputsTracker};
use snark::{MLSnark, SourceMap, SourceType};
use tracing::{error, info};

// #![feature(ascii_char)]
//
// use std::collections::HashMap;
//
// use scalar::scalar;
// use snark::MLSnark;
//
pub mod model;
pub mod subcommands;

pub mod notes;
pub mod scalar;
pub mod snark;
pub mod utils;

pub const SCALE: usize = 1000000;

/// Main crate export. Take a tensor computation and rewrite to snark.
pub fn compile(c: TrainedGraph) -> MLSnark {
  // We set here the weights already. Set input with ::set_input.
  let sc = scalar(c.graph);
  let mut source_map = HashMap::new();
  // set public
  for (i, w_i) in c.weights {
    let little_ids = sc
      .inputs_tracker
      .new_inputs
      .get(&i)
      .unwrap_or_else(|| panic!("Wrong id"));
    for (little_id, v) in little_ids.into_iter().zip(w_i) {
      source_map.insert(*little_id, SourceType::Public(v));
    }
  }
  // set private
  let little_ids = sc
    .inputs_tracker
    .new_inputs
    .get(&c.input_id)
    .unwrap_or_else(|| panic!("Wrong input id"));
  for little_id in little_ids.into_iter() {
    source_map.insert(*little_id, SourceType::Private(None));
  }
  MLSnark {
    graph: sc,
    scale: SCALE,
    source_map: source_map,
    og_input_id: c.input_id,
  }
}

#[cfg(test)]
mod tests {
  // use std::collections::HashMap;

  // use luminal::{graph::Graph, shape::R1};

  // use ark_bls12_381::{Bls12_381, Fr as BlsFr};
  // use ark_groth16::Groth16;
  // use ark_snark::SNARK;

  use ark_bls12_381::Bls12_381;
  use ark_groth16::Groth16;
  use ark_snark::SNARK;

  use crate::{
    compile,
    model::{parse_dataset, run_model, TrainParams},
    snark::{CircuitField, MLSnark},
    utils,
  };

  #[test]
  pub fn test_trained_into_snark_0() -> Result<(), String> {
    utils::init_logging().unwrap();
    let err = |e| format!("{:?}", e).to_string();
    let data = parse_dataset(include_str!("../../data/rp.data").to_string());
    let (_, _model, trained_model) = crate::model::run_model(TrainParams { data, epochs: 1 });
    let mut snark = compile(trained_model);
    let (pk, vk) = snark.make_keys().map_err(err)?;
    // set input
    snark.set_input(vec![0.0; 9]);
    let proof = snark.make_proof(&pk).map_err(err)?;
    let verified = Groth16::<Bls12_381>::verify(&vk, &[CircuitField::from(73)], &proof);
    println!("{:?}", verified);
    // assert!(.unwrap());
    Ok(())
  }
}

use std::{collections::HashMap, vec};

use model::TrainedGraph;
use scalar::{copy_graph_roughly, scalar};
use snark::{CircuitField, MLSnark, SourceType};

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

pub type ScaleT = u64;
pub const SCALE: ScaleT = (2 as u64).pow(f32::MANTISSA_DIGITS);
// pub const SCALE: ScaleT = 100000;

/// Main crate export. Take a tensor computation and rewrite to snark.
pub fn compile(c: &TrainedGraph) -> MLSnark<CircuitField> {
  let graph = copy_graph_roughly(&c.graph);
  let weights = c.weights.clone();
  // We set here the weights already. Set input with ::set_input.
  let sc = scalar(graph);
  let mut source_map = HashMap::new();
  // set public
  for (i, w_i) in weights {
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
    recorded_public_inputs: vec![],
  }
}

#[cfg(test)]
mod tests {

  use crate::{
    compile,
    model::{parse_dataset, TrainParams, TrainedGraph},
    snark::{scaled_float, CircuitField},
    SCALE,
  };
  use ark_bls12_381::Bls12_381;
  use ark_ff::Field;
  use ark_groth16::Groth16;
  use ark_snark::SNARK;
  use itertools::Itertools;

  pub fn test_trained_into_snark(
    mut trained_model: TrainedGraph,
    input: Vec<f32>,
  ) -> Result<(), String> {
    let err = |e| format!("{:?}", e).to_string();
    let scope = crate::utils::init_logging_tests();

    // Public: compile into snark
    let mut snark = compile(&trained_model);
    let (pk, vk) = snark.make_keys().map_err(err)?;

    // Prover: set input and make proof.
    snark.set_input(input.clone());
    let proof = snark.make_proof(&pk).map_err(err)?;
    let public_inputs = snark.recorded_public_inputs.clone();
    // ^ now share public_inputs with Verifier. the vector contains the publically known weights (available already in the Public step) and the evaluation result of the model, which the prover shares, as otherwise he's not proving anything.

    // Verifier: verify the proof
    let verified = Groth16::<Bls12_381>::verify(&vk, &public_inputs, &proof);

    // God: and compare the results obtained
    let snark_eval_result = snark.get_evaluation_result(); // this really just is public_inputs[-1], a publicly known result of the circuit
    let model_eval_res_float = trained_model.evaluate(input)[0];
    let model_eval_result: CircuitField = scaled_float(model_eval_res_float, SCALE);
    tracing::info!("{:?} {:?}", snark_eval_result, model_eval_result);

    let diff = (snark_eval_result - model_eval_result)
      .square()
      .le(&scaled_float(0.01, SCALE));

    assert!(verified == Ok(true), "Proof is verified");
    assert!(
      diff,
      "The snark evaluates to the correct result (~ float precision)"
    );
    tracing::info!("evaluated the model to {:?}, which is represented by a field element {:?}. Also evaluated the snark to a field element {:?}. The two results are within 0.01 float margin. Verifier correctly verified the proof that snark evaluates to that value.", model_eval_res_float, model_eval_result, snark_eval_result);

    drop(scope);
    Ok(())
  }

  // #[ignore]
  #[test]
  pub fn test_trained_into_snark_0() -> Result<(), String> {
    // See the model shape at https://dreampuf.github.io/GraphvizOnline/#digraph%20%7B%0A%20%20%20%200%20%5B%20label%20%3D%20%22Weight%20Load%20%7C%200%22%20%5D%0A%20%20%20%201%20%5B%20label%20%3D%20%22Tensor%20Load%20%7C%201%22%20%5D%0A%20%20%20%202%20%5B%20label%20%3D%20%22Mul%20%7C%202%22%20%5D%0A%20%20%20%203%20%5B%20label%20%3D%20%22SumReduce(2)%20%7C%203%22%20%5D%0A%20%20%20%200%20-%3E%202%20%5B%20%20%5D%0A%20%20%20%201%20-%3E%202%20%5B%20%20%5D%0A%20%20%20%202%20-%3E%203%20%5B%20%20%5D%0A%7D%0A
    tracing::info!("linear layer, data A");
    let data = parse_dataset(include_str!("../../data/rp.data").to_string());
    let trained_model = crate::model::tiny_model::run_model(TrainParams { data, epochs: 1 });
    let input = (0..9).map(|x| f32::from(x as i16)).collect_vec();
    test_trained_into_snark(trained_model, input)
  }

  // #[ignore]
  #[test]
  pub fn test_trained_into_snark_1() -> Result<(), String> {
    tracing::info!("linear layer, data B");
    let data = parse_dataset(include_str!("../../data/rp.data").to_string());
    let trained_model = crate::model::tiny_model::run_model(TrainParams { data, epochs: 1 });
    let input = (9..18).map(|x| f32::from(x as i16)).collect_vec();
    test_trained_into_snark(trained_model, input)
  }

  // #[ignore]
  #[test]
  pub fn test_trained_into_snark_2() -> Result<(), String> {
    // see the model shape at https://dreampuf.github.io/GraphvizOnline/#digraph%20%7B%0A%20%20%20%200%20%5B%20label%20%3D%20%22Weight%20Load%20%7C%200%22%20%5D%0A%20%20%20%201%20%5B%20label%20%3D%20%22Weight%20Load%20%7C%201%22%20%5D%0A%20%20%20%202%20%5B%20label%20%3D%20%22Tensor%20Load%20%7C%202%22%20%5D%0A%20%20%20%203%20%5B%20label%20%3D%20%22Mul%20%7C%203%22%20%5D%0A%20%20%20%204%20%5B%20label%20%3D%20%22SumReduce(2)%20%7C%204%22%20%5D%0A%20%20%20%205%20%5B%20label%20%3D%20%22Constant(0.0)%20%7C%205%22%20%5D%0A%20%20%20%206%20%5B%20label%20%3D%20%22LessThan%20%7C%206%22%20%5D%0A%20%20%20%207%20%5B%20label%20%3D%20%22Mul%20%7C%207%22%20%5D%0A%20%20%20%208%20%5B%20label%20%3D%20%22LessThan%20%7C%208%22%20%5D%0A%20%20%20%209%20%5B%20label%20%3D%20%22Constant(-1.0)%20%7C%209%22%20%5D%0A%20%20%20%2010%20%5B%20label%20%3D%20%22Mul%20%7C%2010%22%20%5D%0A%20%20%20%2011%20%5B%20label%20%3D%20%22Constant(1.0)%20%7C%2011%22%20%5D%0A%20%20%20%2012%20%5B%20label%20%3D%20%22Add%20%7C%2012%22%20%5D%0A%20%20%20%2013%20%5B%20label%20%3D%20%22Mul%20%7C%2013%22%20%5D%0A%20%20%20%2014%20%5B%20label%20%3D%20%22Add%20%7C%2014%22%20%5D%0A%20%20%20%2015%20%5B%20label%20%3D%20%22Mul%20%7C%2015%22%20%5D%0A%20%20%20%2016%20%5B%20label%20%3D%20%22SumReduce(2)%20%7C%2016%22%20%5D%0A%20%20%20%200%20-%3E%203%20%5B%20%20%5D%0A%20%20%20%201%20-%3E%2015%20%5B%20%20%5D%0A%20%20%20%202%20-%3E%203%20%5B%20%20%5D%0A%20%20%20%203%20-%3E%204%20%5B%20%20%5D%0A%20%20%20%204%20-%3E%208%20%5B%20%20%5D%0A%20%20%20%204%20-%3E%206%20%5B%20%20%5D%0A%20%20%20%204%20-%3E%2013%20%5B%20%20%5D%0A%20%20%20%205%20-%3E%208%20%5B%20%20%5D%0A%20%20%20%205%20-%3E%207%20%5B%20%20%5D%0A%20%20%20%205%20-%3E%206%20%5B%20%20%5D%0A%20%20%20%206%20-%3E%207%20%5B%20%20%5D%0A%20%20%20%207%20-%3E%2014%20%5B%20%20%5D%0A%20%20%20%208%20-%3E%2010%20%5B%20%20%5D%0A%20%20%20%209%20-%3E%2010%20%5B%20%20%5D%0A%20%20%20%2010%20-%3E%2012%20%5B%20%20%5D%0A%20%20%20%2011%20-%3E%2012%20%5B%20%20%5D%0A%20%20%20%2012%20-%3E%2013%20%5B%20%20%5D%0A%20%20%20%2013%20-%3E%2014%20%5B%20%20%5D%0A%20%20%20%2014%20-%3E%2015%20%5B%20%20%5D%0A%20%20%20%2015%20-%3E%2016%20%5B%20%20%5D%0A%7D%0A
    tracing::info!("linear layer into ReLU, data A");
    let data = parse_dataset(include_str!("../../data/rp.data").to_string());
    let trained_model = crate::model::lessthan_model::run_model(TrainParams { data, epochs: 1 });
    let input = (0..9).map(|x| f32::from(x as i16)).collect_vec();
    test_trained_into_snark(trained_model, input)
  }

  // #[ignore]
  #[test]
  pub fn test_trained_into_snark_3() -> Result<(), String> {
    tracing::info!("linear layer into ReLU, data B");
    let data = parse_dataset(include_str!("../../data/rp.data").to_string());
    let trained_model = crate::model::lessthan_model::run_model(TrainParams { data, epochs: 1 });
    let input = (9..18).map(|x| f32::from(x as i16)).collect_vec();
    test_trained_into_snark(trained_model, input)
  }

  #[test]
  pub fn test_trained_into_snark_4() -> Result<(), String> {
    tracing::info!("linear layer into ReLU, data C");
    let data = parse_dataset(include_str!("../../data/rp.data").to_string());
    let trained_model = crate::model::lessthan_model::run_model(TrainParams { data, epochs: 1 });
    let input: Vec<f32> = [1.001231212412512, 0.3141512, 8910395712741e-10, 136213e12, 7819421e-4, 71289401e18, 9801721e-14, 0.763612199124, 0.12199124].to_vec();
    test_trained_into_snark(trained_model, input)
  }

  #[ignore = "runs for too long"]
  #[test]
  pub fn test_trained_into_snark_5() -> Result<(), String> {
    let data = parse_dataset(include_str!("../../data/rp.data").to_string());
    let trained_model = crate::model::medium_model::run_model(TrainParams { data, epochs: 1 });
    let input = (0..9).map(|x| f32::from(x as i16)).collect_vec();
    test_trained_into_snark(trained_model, input)
  }
}

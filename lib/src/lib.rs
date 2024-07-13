use std::collections::HashMap;

use itertools::Itertools;
use luminal::prelude::NodeIndex;
use model::{get_weights, Model};
use snark::MLSnark;
use scalar::scalar;

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


/// Main crate export. Take a tensor computation and rewrite to snark.
pub fn compile<F>(c: luminal::graph::Graph, m: Model) -> MLSnark {
  // TODO: BIG TODO: care about source map, need to record it from the beginning.
  // TODO: not mutate c
  let weights = model::get_weights(&c, &m);
  let weights: HashMap<NodeIndex, Option<Vec<f32>>> = weights.iter().map(|(k, v)| {(k.to_owned(), Some(v.to_owned()))}).collect();
  let sc = scalar(c);
  MLSnark {
    graph: sc,
    scale: 200,
    private_inputs: weights,
  }
}

// #[cfg(test)]
// mod tests {
//   use std::collections::HashMap;
//
//   use luminal::{graph::Graph, shape::R1};
//
//   use ark_bls12_381::{Bls12_381, Fr as BlsFr};
//   use ark_groth16::Groth16;
//   use ark_snark::SNARK;
//
//   use crate::{compile, snark::MLSnark};
//
//   // THIS doesnt work because sources map uncared for
//   // But this is how we'd use it
//   #[test]
//   pub fn test0() {
//     let mut cx = Graph::new();
//     let a = cx.tensor::<R1<3>>().set(vec![1.0, 2.0, 3.0]);
//     let b = cx.tensor::<R1<3>>().set(vec![1.0, 2.0, 3.0]);
//     let d = cx.tensor::<R1<3>>().set(vec![0.0, 1.0, 0.0]);
//
//     let c = ((a + b) + d).retrieve();
//
//     let snark = compile(cx);
//
//     let rng = &mut ark_std::test_rng();
//
//     // generate the setup parameters
//     let (pk, vk) = Groth16::<Bls12_381>::circuit_specific_setup(snark, rng).unwrap();
//
//     // calculate the proof by passing witness variable value
//     let proof1 = Groth16::<Bls12_381>::prove(
//         &pk,
//         MLSnark {
//           source_nodes_map : todo!("Provide private inputs here"),
//           graph: todo!("Need to fix ownership - cant move graph, so probably we should forget Graph in MlSnark already")
//         },
//         rng,
//         );
//   }
// }

// todo: abstract away the training loop. split from the lib crate
pub mod fixed_weights;
pub mod lessthan_model;
pub mod medium_model;
pub mod tiny_model;

use luminal::{
  graph::Graph,
  module::{InitModule, Module},
  prelude::NodeIndex,
  shape::R1,
};
use luminal_training::Autograd;
pub use medium_model::*;

pub type Dataset = (Vec<[f32; 9]>, Vec<f32>);

pub struct TrainingParams {
  pub data: (InputsVec, OutputsVec),
  pub epochs: usize,
  // pub model: Model,
}

pub fn from_weights(weights: Vec<(u32, Vec<f32>)>) -> GraphForSnark {
  /* We assume that the entire process of creating the network is deterministic */
  let mut cx = Graph::new();
  let model = <Model>::initialize(&mut cx);
  let input = cx.tensor::<R1<9>>();
  let output = model.forward(input).retrieve();
  let target = cx.tensor::<R1<1>>();
  let loss = luminal_training::mse_loss(output, target).retrieve();
  let parameters = luminal::module::params(&model);
  cx.compile(Autograd::new(&parameters, loss), ());
  let input_id = input.id;
  GraphForSnark {
    graph: cx,
    input_id,
    weights: weights
      .iter()
      .map(|(id, tensor)| (NodeIndex::from(id.clone()), tensor.clone()))
      .collect(),
  }
}

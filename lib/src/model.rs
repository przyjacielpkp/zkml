pub mod fixed_weights;
pub mod training;
pub mod types;
pub mod utils;

use luminal::prelude::NodeIndex;

pub use training::*;
pub use types::*;
pub use utils::*;

pub fn from_weights(weights: Vec<(u32, Vec<f32>)>) -> GraphForSnark {
  /* We assume that the entire process of creating the network is deterministic */
  let mock_graph = run_model(TrainingParams {
    data: (Vec::new(), Vec::new()),
    epochs: 0,
  });
  GraphForSnark {
    weights: weights
      .iter()
      .map(|(id, tensor)| (NodeIndex::from(*id), tensor.clone()))
      .collect(),
    ..mock_graph.graph
  }
}

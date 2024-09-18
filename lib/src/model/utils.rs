use luminal::prelude::*;
use std::{convert::TryInto, path::Path};

use crate::scalar::copy_graph_roughly;

use super::{InputsVec, OutputsVec};

/// Contains everything needed to define the snark: the ml graph but without the gradients, trained weights and indexes.
/// Note: this is quite a specific and frankly poor interface between training and snark synthesiz, so don't take it as engraved in stone.
#[derive(Debug)]
pub struct GraphForSnark {
  // the initial ml computation graph, without gradients
  pub graph: Graph,
  pub input_id: NodeIndex,
  pub weights: Vec<(NodeIndex, Vec<f32>)>,
}

impl GraphForSnark {
  pub fn copy_graph_roughly(&self) -> Self {
    let (g, remap) = copy_graph_roughly(&self.graph);
    GraphForSnark {
      graph: g,
      input_id: remap[&self.input_id],
      weights: self
        .weights
        .iter()
        .map(|(a, b)| (remap[a], b.clone()))
        .collect(),
    }
  }
}

/// Contains everything needed to define a snark and also evaluate the model.
/// Note: this is quite a specific and frankly poor interface between training and snark synthesiz, so don't take it as engraved in stone.
///       Generally: this is graph + some stuff recorded to evaluate it on input.
#[derive(Debug)]
pub struct TrainedGraph {
  /// the original ml computation graph, without gradients + input id + trained weights
  pub graph: GraphForSnark,
  // below are needed to evaluate the model to compare result against a snark derived from GraphForSnark:
  pub cx: Graph,
  /// full trained graph for evaluation, the above "graph" is similar but without gradients
  pub cx_weights: Vec<(NodeIndex, Vec<f32>)>, // needed for evaluation, mostly tests. redundant a bit
  pub cx_input_id: NodeIndex,  // needed for evaluation, mostly tests
  pub cx_target_id: NodeIndex, // needed for evaluation, mostly tests
  pub cx_output_id: NodeIndex,
}

impl TrainedGraph {
  pub fn evaluate(&mut self, input_data: Vec<f32>) -> Vec<f32> {
    self.cx.get_op_mut::<Function>(self.cx_input_id).1 =
      Box::new(move |_| vec![Tensor::new(input_data.clone())]);
    self.cx.get_op_mut::<Function>(self.cx_target_id).1 =
      Box::new(move |_| vec![Tensor::new(vec![0.0])]); // doesnt matter
    let weights = self.cx_weights.clone();
    for (a, b) in weights {
      self.cx.get_op_mut::<Function>(a).1 = Box::new(move |_| vec![Tensor::new(b.clone())]);
    }
    self.cx.execute();
    let d = self
      .cx
      .get_tensor_ref(self.cx_output_id, 0)
      .unwrap()
      .clone()
      .downcast_ref::<Vec<f32>>()
      .unwrap()
      .clone();
    d
  }
}

pub fn parse_dataset(content: String) -> (InputsVec, OutputsVec) {
  let mut x: InputsVec = Vec::new();
  let mut y: OutputsVec = Vec::new();
  for line in content.lines() {
    let parts: Vec<_> = line
      .split_whitespace()
      .map(|val| val.parse::<f32>().unwrap())
      .collect();
    assert_eq!(parts.len(), super::INPUT_DIMENSION + 1);
    x.push(parts[0..super::INPUT_DIMENSION].try_into().unwrap());
    y.push(parts[super::INPUT_DIMENSION] / 2.0 - 1.0);
  }
  (x, y)
}

pub fn read_dataset(path: &Path) -> (InputsVec, OutputsVec) {
  let content: String = match std::fs::read_to_string(path) {
    Ok(content) => content,
    Err(e) => panic!("Failed to read file {:?}: {}", path, e),
  };
  parse_dataset(content)
}

pub fn split_dataset(
  x: InputsVec,
  y: OutputsVec,
  ratio: f32,
) -> (InputsVec, InputsVec, OutputsVec, OutputsVec) {
  assert_eq!(x.len(), y.len());
  let splitting_point = (x.len() as f32 * ratio) as usize;

  let (x_train, x_test) = x.split_at(splitting_point);
  let (y_train, y_test) = y.split_at(splitting_point);

  (
    x_train.to_vec(),
    x_test.to_vec(),
    y_train.to_vec(),
    y_test.to_vec(),
  )
}

pub fn normalize_data(x: InputsVec) -> InputsVec {
  let mut mins: [f32; 9] = [f32::INFINITY; 9];
  let mut maxs: [f32; 9] = [-f32::INFINITY; 9];

  for a in x.iter() {
    for i in 0..9 {
      mins[i] = f32::min(mins[i], a[i]);
      maxs[i] = f32::min(maxs[i], a[i]);
    }
  }

  let mut xp: InputsVec = Vec::new();
  for a in x.iter() {
    let mut ap: [f32; 9] = [0 as f32; 9];
    for i in 0..9 {
      ap[i] = (a[i] - mins[i]) / (maxs[i] - mins[i]);
    }
    xp.push(ap);
  }
  xp
}

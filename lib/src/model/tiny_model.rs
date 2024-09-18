use std::iter::zip;

use luminal::prelude::*;
use luminal_nn::Linear;
use luminal_training::{mse_loss, sgd_on_graph, Autograd};

use super::{GraphForSnark, InputsVec, OutputsVec};
use crate::scalar::copy_graph_roughly;

use super::{TrainedGraph, TrainingParams};

pub type Model = (Linear<9, 1>,);

pub fn run_model(training_params: TrainingParams) -> TrainedGraph {
  let dataset: (InputsVec, OutputsVec) = training_params.data;
  let epochs = training_params.epochs;
  // Setup gradient graph
  let mut cx = Graph::new();
  let model = <Model>::initialize(&mut cx);
  let input = cx.tensor::<R1<9>>();
  let output = model.forward(input).retrieve();

  // record graph without gradients.
  let (cx_og, remap) = copy_graph_roughly(&cx);
  let input_id = remap[&input.id];

  let target = cx.tensor::<R1<1>>();
  let loss = mse_loss(output, target).retrieve();
  let weights = params(&model);

  let grads = cx.compile(Autograd::new(&weights, loss), ());
  let (new_weights, lr) = sgd_on_graph(&mut cx, &weights, &grads);
  cx.keep_tensors(&new_weights);
  cx.keep_tensors(&weights);
  lr.set(5e-3);

  let (mut loss_avg, mut acc_avg) = (ExponentialAverage::new(1.0), ExponentialAverage::new(0.0));
  let start = std::time::Instant::now();

  let (x, y) = dataset;
  let (training_x, _, training_y, _) = super::split_dataset(x, y, 0.8);
  let training_x = super::normalize_data(training_x);
  let mut iter = 0;
  for _ in 0..epochs {
    for (x, y) in zip(training_x.iter(), training_y.iter()) {
      let answer = [*y];
      input.set(*x);
      target.set(answer);

      cx.execute();
      transfer_data_same_graph(&new_weights, &weights, &mut cx);
      loss_avg.update(loss.data()[0]);
      loss.drop();
      // println!("{:}, {:}", output.data()[0], answer[0]);
      acc_avg.update(
        output
          .data()
          .into_iter()
          .zip(answer)
          .filter(|(a, b)| (a - b).abs() < 0.5)
          .count() as f32,
      );
      tracing::info!("{:?}", output.data());
      output.drop();
      iter += 1;
    }
  }

  if iter > 0 {
    tracing::info!("Finished in {iter} iterations");
    tracing::info!(
      "Took {:.2}s, {:.2}µs / iter",
      start.elapsed().as_secs_f32(),
      start.elapsed().as_micros() / iter
    );
  }

  let cx_weights_vec: Vec<(NodeIndex, Vec<f32>)> = weights
    .into_iter()
    .map(|a| {
      (
        a,
        cx.tensors
          .get(&(a, 0 /* assuming single output */))
          .map(|val| val.downcast_ref::<Vec<f32>>().unwrap())
          .unwrap_or(&Vec::new())
          .clone()
          .into_iter()
          .collect(),
      )
    })
    .collect();
  let weights = cx_weights_vec
    .iter()
    .map(|(a, b)| (remap[a], b.clone()))
    .collect();

  TrainedGraph {
    graph: GraphForSnark {
      graph: cx_og,
      weights,
      input_id,
    },
    cx,
    cx_weights: cx_weights_vec,
    cx_output_id: output.id,
    cx_input_id: input.id,
    cx_target_id: target.id,
  }
}

pub struct ExponentialAverage {
  beta: f32,
  moment: f32,
  pub value: f32,
  t: i32,
}

impl ExponentialAverage {
  pub fn new(initial: f32) -> Self {
    ExponentialAverage {
      beta: 0.999,
      moment: 0.,
      value: initial,
      t: 0,
    }
  }
}

impl ExponentialAverage {
  pub fn update(&mut self, value: f32) {
    self.t += 1;
    self.moment = self.beta * self.moment + (1. - self.beta) * value;
    // bias correction
    self.value = self.moment / (1. - f32::powi(self.beta, self.t));
  }

  pub fn reset(&mut self) {
    self.moment = 0.;
    self.value = 0.0;
    self.t = 0;
  }
}

use std::iter::zip;

use luminal::prelude::*;
use luminal_nn::{Linear, ReLU};
use luminal_training::{mse_loss, sgd_on_graph, Autograd};
use tracing::info;

use crate::{
  model::{normalize_data, split_dataset, ExponentialAverage, GraphForSnark, InputsVec, OutputsVec},
  scalar::copy_graph_roughly,
};

use super::{TrainParams, TrainedGraph};

pub type Model = (Linear<9, 2>, ReLU, Linear<2, 1>);

pub fn run_model(train_params: TrainParams) -> TrainedGraph {
  let dataset: (InputsVec, OutputsVec) = train_params.data;
  let epochs = train_params.epochs;
  // Setup gradient graph
  let mut cx = Graph::new();
  let model = <Model>::initialize(&mut cx);
  let mut input = cx.tensor::<R1<9>>();
  let mut output = model.forward(input).retrieve();

  // todo: remove x=n
  // cx.display();
  cx.compile(
    GenericCompiler::default(),
    (
      &mut input,
      &mut output,
    ),
  );

  // cx.display();
  // cx.display_shapes();
  // record graph without gradients. assuming nodeids dont change in Autograd::compile
  let (cx_og, remap) = copy_graph_roughly(&cx);
  let input_id = input.id;

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
  // let EPOCHS = 20;

  let (X, Y) = dataset;
  let (X_train, _x_test, y_train, _y_test) = split_dataset(X, Y, 0.8);
  let X_train = normalize_data(X_train);
  let mut iter = 0;
  for _ in 0..epochs {
    for (x, y) in zip(X_train.iter(), y_train.iter()) {
      let answer = [y.to_owned()];
      input.set(x.to_owned());
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
      info!("{:?}", output.data());
      output.drop();
      // println!(
      //   "Iter {iter} Loss: {:.2} Acc: {:.2}",
      //   loss_avg.value, acc_avg.value
      // );
      iter += 1;
    }
  }
  println!("Finished in {iter} iterations");
  println!(
    "Took {:.2}s, {:.2}µs / iter",
    start.elapsed().as_secs_f32(),
    start.elapsed().as_micros() / iter
  );
  // cx.display();
  let weights_vec = weights
    .into_iter()
    .map(|a| {
      (
        remap[&a],
        cx.tensors
          .get(&(a, 0 /* assuming single output */))
          .unwrap()
          .downcast_ref::<Vec<f32>>()
          .unwrap()
          .clone()
          .into_iter()
          .collect(),
      )
    })
    .collect();
  TrainedGraph {
    graph : GraphForSnark {
      graph: cx_og,
      weights: weights_vec,
      input_id,
    },
    cx: cx,
    cx_output_id: output.id,
    cx_input_id: input.id,
    cx_target_id: target.id,
  }
}

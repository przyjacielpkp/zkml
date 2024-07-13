use std::{
  any::Any,
  collections::HashMap,
  convert::TryInto,
  fs::{self, File},
  iter::zip,
  ops::Deref,
  path::Path,
};

use dfdx::shapes::Const;
use luminal::prelude::*;
use luminal_nn::{Linear, ReLU};
use luminal_training::{mse_loss, sgd_on_graph, Autograd};
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences};

const FILE_PATH: &str = "data/rp.data";

pub type InputsVec = Vec<[f32; 9]>;
pub type OutputsVec = Vec<f32>;

pub type Model = (Linear<9, 16>, ReLU, Linear<16, 16>, ReLU, Linear<16, 1>);

pub fn read_dataset(path: &Path) -> (InputsVec, OutputsVec) {
  let content: Vec<String> = fs::read_to_string(path)
    // todo: error handling
    .unwrap()
    .lines()
    .map(String::from)
    .collect();

  // todo: why no csv?
  let mut x: InputsVec = Vec::new();
  let mut y: OutputsVec = Vec::new();
  for line in content {
    let mut parts: Vec<&str> = line.split(" ").collect();
    parts.retain(|&a| a != "");
    let parts: OutputsVec = parts.iter().map(|a| a.parse::<f32>().unwrap()).collect();
    let len = parts.len();
    x.push(parts[0..len - 1].try_into().unwrap());
    if parts[len - 1] == 2.0 {
      y.push(0.);
    } else {
      y.push(1.);
    }
  }
  (x, y)
}

pub fn split_dataset(
  x: InputsVec,
  y: OutputsVec,
  ratio: f32,
) -> (InputsVec, InputsVec, OutputsVec, OutputsVec) {
  let len = x.len();
  let len_short = (len as f32 * ratio) as usize;
  let x_train = x[0..len_short].to_vec();
  let x_test = x[len_short..len - 1].to_vec();
  let y_train = y[0..len_short].to_vec();
  let y_test = y[len_short..len - 1].to_vec();

  (x_train, x_test, y_train, y_test)
}

pub fn normalize_data(x: InputsVec) -> InputsVec {
  let mut mins: [f32; 9] = [11 as f32; 9];
  let mut maxs: [f32; 9] = [-1 as f32; 9];

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

pub fn get_weights(graph: &Graph, model: &Model) -> Vec<(NodeIndex, Vec<f32>)> {
  let weights_indices = params(model);
  weights_indices
    .iter()
    .map(|index| {
      (
        *index,
        graph
          .tensors
          .get(&(*index, 0))
          .unwrap()
          .downcast_ref::<Vec<f32>>()
          .unwrap()
          .clone(),
      )
    })
    .collect()
}

pub fn get_accuracy(
  cx: &mut Graph,
  input: GraphTensor<R1<9>>,
  output: GraphTensor<R1<1>>,
  X: &InputsVec,
  y: &OutputsVec,
) -> f32 {
  let mut cnt: u32 = 0;
  for (x, ans) in zip(X, y) {
    input.set(x.to_owned());
    cx.execute();
    if output.data()[0] == ans.to_owned() {
      cnt += 1;
    }
    output.drop();
  }
  cnt as f32 / y.len() as f32
}

pub struct TrainingParams {
  pub data: (InputsVec, OutputsVec),
  pub epochs: usize,
  // pub lr: f32,
  // pub batch_size: u32,
}

pub struct TrainedGraph {
  pub graph: Graph,
  pub input_id: NodeIndex,
  pub weights: Vec<(NodeIndex, Vec<f32>)>,
}

pub fn run_model(training_params: TrainingParams) -> TrainedGraph {
  let dataset: (InputsVec, OutputsVec) = training_params.data;
  let epochs = training_params.epochs;
  // Setup gradient graph
  let mut cx = Graph::new();
  let model = <Model>::initialize(&mut cx);
  let mut input = cx.tensor::<R1<9>>();
  let mut target = cx.tensor::<R1<1>>();
  let mut output = model.forward(input).retrieve();
  let mut loss = mse_loss(output, target).retrieve();

  let mut weights = params(&model);
  cx.display();
  // record graph without gradients. assuming nodeids dont change in Autograd::compile
  let cx_og = copy_graph_roughly(&cx);

  let grads = cx.compile(Autograd::new(&weights, loss), ());
  let (mut new_weights, lr) = sgd_on_graph(&mut cx, &weights, &grads);
  cx.keep_tensors(&new_weights);
  cx.keep_tensors(&weights);
  lr.set(5e-3);

  #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
  cx.compile(
    GenericCompiler::default(),
    (
      &mut input,
      &mut target,
      &mut loss,
      &mut output,
      &mut weights,
      &mut new_weights,
    ),
  );

  let (mut loss_avg, mut acc_avg) = (ExponentialAverage::new(1.0), ExponentialAverage::new(0.0));
  let start = std::time::Instant::now();
  // let epochs = 20;

  let (X, Y) = dataset;
  let (X_train, X_test, y_train, y_test) = split_dataset(X, Y, 0.8);
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
      println!("{:}, {:}", output.data()[0], answer[0]);
      acc_avg.update(
        output
          .data()
          .into_iter()
          .zip(answer)
          .filter(|(a, b)| (a - b).abs() < 0.5)
          .count() as f32,
      );
      output.drop();
      println!(
        "Iter {iter} Loss: {:.2} Acc: {:.2}",
        loss_avg.value, acc_avg.value
      );
      iter += 1;
    }
  }
  println!("Finished in {iter} iterations");
  println!(
    "Took {:.2}s, {:.2}µs / iter",
    start.elapsed().as_secs_f32(),
    start.elapsed().as_micros() / iter
  );
  cx.display();

  TrainedGraph {
    graph: cx_og,
    input_id: input.id,
    weights: get_weights(&cx, &model),
  }
}

pub struct ExponentialAverage {
  beta: f32,
  moment: f32,
  pub value: f32,
  t: i32,
}

impl ExponentialAverage {
  fn new(initial: f32) -> Self {
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

// copies things that are relevant. very much not exact copy
pub fn copy_graph_roughly(src: &Graph) -> Graph {
  let mut g = Graph::new();
  let mut map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
  for x in src.node_indices() {
    let n = if src.check_node_type::<Add>(x) {
      g.add_op(Add {}).finish()
    } else if src.check_node_type::<Mul>(x) {
      g.add_op(Mul {}).finish()
    } else if src.check_node_type::<Function>(x) {
      g.add_op(Function(
        "Load".to_string(),
        Box::new(|_| panic!("dont run")),
      ))
      .finish()
    } else if src.check_node_type::<Recip>(x) {
      g.add_op(Recip {}).finish()
    } else if src.check_node_type::<MaxReduce>(x) {
      let op = src.get_op::<MaxReduce>(x);
      g.add_op(MaxReduce(op.0)).finish()
    } else if src.check_node_type::<SumReduce>(x) {
      let op = src.get_op::<SumReduce>(x);
      g.add_op(SumReduce(op.0)).finish()
    } else if src.check_node_type::<Constant>(x) {
      let op = src.get_op::<Constant>(x);
      g.add_op(Constant(op.0.clone(), op.1)).finish()
    } else {
      panic!("Unknown node type")
    };
    map.insert(x, n);
  }
  for e in src.edge_references() {
    g.add_edge(e.source(), e.target(), e.weight().clone());
  }
  g
}


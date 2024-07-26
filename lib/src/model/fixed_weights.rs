
use luminal::prelude::*;
use luminal_nn::{Linear, ReLU};
use petgraph::Direction::Outgoing;

use crate::{
  model::GraphForSnark,
  scalar::copy_graph_roughly,
};

use super::TrainedGraph;

pub type Model = (Linear<3, 2>, ReLU, Linear<2, 1>);

// from scalar
fn get_own_size(x: NodeIndex, gg: &Graph) -> usize {
  let get_own_shape = |x, gg: &Graph| {
    // reasonably we expect one of two cases: there is some outgoing edge OR it is a retrieval node
    if let Some(w) = gg.to_retrieve.get(&x) {
      w.clone().1
    } else {
      match gg
        .edges_directed(x.clone(), Outgoing)
        .filter_map(|e| e.weight().as_data())
        .next()
      {
        Some((_, _, shape)) => shape,
        None => {
          panic!("A node has no outgoing edges and is not a retrieval node.")
        }
      }
    }
  };
  // assuming (and we have to) a staticly known shape
  match get_own_shape(x, gg).n_physical_elements().to_usize() {
    Some(n) => n,
    None => {
      panic!("Node's output shape is not static.")
    }
  }
}

pub fn run_model() -> TrainedGraph {
  let mut cx = Graph::new();
  let model = <Model>::initialize(&mut cx);
  let mut input = cx.tensor::<R1<3>>();
  let mut output = model.forward(input).retrieve();

  cx.compile(GenericCompiler::default(), (&mut input, &mut output));

  // cx.display();
  // record graph without gradients. assuming nodeids dont change in Autograd::compile
  let (cx_og, remap) = copy_graph_roughly(&cx);
  let input_id = input.id;

  let target = cx.tensor::<R1<1>>();
  // let loss = mse_loss(output, target).retrieve();
  let weights = params(&model);

  let mut j = 0;
  let mut weights_vec = vec![];
  let mut cx_weights_vec = vec![];
  for a in weights {
    let len = get_own_size(a, &cx);
    let vect: Vec<f32> = (j..(j + len)).map(|x| x as f32).collect();
    weights_vec.push((remap[&a], vect.clone()));
    cx_weights_vec.push((a, (j..(j + len)).map(|x| x as f32).collect()));
    j += len;
  }
  TrainedGraph {
    graph: GraphForSnark {
      graph: cx_og,
      weights: weights_vec,
      input_id,
    },
    cx: cx,
    cx_weights: cx_weights_vec,
    cx_output_id: output.id,
    cx_input_id: input.id,
    cx_target_id: target.id,
    // cx_target_id: output.id, // <- whatever
  }
}

///
/// Rewrite the ml source graph to scalarised version, where tensor operations are replaced with a multiplicity of scalar operations.
///  

/// Note: tensors may have dynamic dimensions.
/// Solution 1: assume not
// Problem: What about nodes that output multiple values? Add, Mul, LessThan, ReduceAdd - are not like that right?
use luminal::graph::Graph;

use std::{collections::HashMap, error::Error, fs::File, io::Write};

use itertools::Itertools;
use petgraph::{
  visit::{EdgeRef, IntoNodeIdentifiers},
  Direction::{Incoming, Outgoing},
};
use tracing::{debug, info, instrument, warn};

use luminal::{
  op::{Constant, ConstantValue, InputTensor, Operator},
  prelude::*,
  shape::Shape,
};

/// Asserts (in non-strictly-typed way) that all input tensors are single values.
#[derive(Debug)]
pub struct ScalarGraph {
  pub graph: Graph,
}

/// Rewrite the static tensor computation to scalar computation.
pub fn scalar(mut cx: Graph) -> ScalarGraph {
  // TODO: unfortunetely original cx is destroyed in the process
  // let mut cx1 = (&cx).clone().clone();
  // we dont care about remap for now
  let mut remap: Vec<NodeIndex> = vec![];
  let ((), ()) = cx.compile(ScalarCompiler::default(), &mut remap);
  ScalarGraph { graph: cx }
}

#[derive(Debug, Default)]
pub struct UniformOutShapes;

/// Kinda obsolete
impl Compiler for UniformOutShapes {
  type Output = ();
  #[instrument(level = "debug", skip(ids))]
  fn compile<T: ToIdsMut>(&self, graph: &mut Graph, ids: T) -> Self::Output {
    // For every node substitute as many copies of it as there are distinct outgoing shapes.
    // Connect the new nodes to the target nodes correspondingly wrt shapes.

    // Assuming : output_in = 0
    // Shapes could actually be different

    debug!("Assuming from every node all outgoing edges are of same shape and output_in.");
    for node in graph.graph.node_indices() {
      let all_equal = graph
        .graph
        .edges_directed(node, Outgoing)
        .filter_map(|e| e.weight().as_data().map(|w| (w.1 /* hope equal 0 */, w.2)))
        .all_equal();
      assert!(
        all_equal,
        "All outgoing edges of a node must have the same shape."
      )
    }
  }
}

pub type ScalarCompiler = (UniformOutShapes, Scalarize);

#[derive(Debug, Default, Clone)]
/// In the scalar graph used for source nodes no matter they original Op.
pub struct InputOp {}

impl Operator for InputOp {
  fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
    panic!("InputOp: We wont be evaluating it either way")
  }
}

#[derive(Debug, Default)]
pub struct Scalarize;

impl Compiler for Scalarize {
  type Output = ();

  // THIS-WORKS

  #[instrument(level = "debug", name = "compile", skip(_ids))]
  /// Start from the sinks in graph and go backwards.
  /// Look at a node - assume all its outgoing shapes are the same (due to UniformOutShapes).
  /// We want to rewrite it to many little nodes.
  /// From previous steps the outgoing edges are already multiplied into shape many edges.
  /// We want to create shape many little nodes with outputs (and as many as needed nodes to implement the rest of the circuit).
  /// We connect the outgoing edges to corresponding little nodes using indices like with tensors.
  /// We create edges connecting our little nodes to source nodes. For every source there will source's shape many edges going from that source.
  fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut _ids: T) {
    // Assumes that all outgoing edges have same shape from a given node. NOTE: why? not needed once realized physical shape is always going to be same for single output.
    // FIX: ^ Not true.

    // Q: do inefficient but simpler with Looped<(AddCompile, MulCompile)> etc and pattern matching
    //    or efficiently in a single for loop in toposort order (and meticoulous manual pattern matching)
    // A: option 2, because cant do 1 efficiently

    // TODO: What about ops returning many tensors? (no prim ops right?)
    // Problem: We decide little nodes amount based on outgoing shape, assuming there's one tensor produced.

    // mark retrieve nodes
    let mark_retrieve = |x: &NodeIndex, new_xs: Vec<_>, g: &mut Graph| {
      if let Some(w) = g.to_retrieve.get(x) {
        for new_x in new_xs {
          // let new_x : NodeIndex = new_x;
          g.to_retrieve.insert(
            new_x,
            (
              0, /* this probably refers to output index in Vec<Tensor> */
              R0::to_tracker(),
            ),
          );
        }
      }
    };

    let get_own_size = |x, gg: &Graph| {
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
              panic!("Add node has no outgoing edges and is not a retrieval node.")
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
    };

    // We split node into multiple nodes instead.
    // This helper creates the little nodes all with same Op.
    fn make_nodes<T: Operator + 'static + Clone>(
      size: usize,
      op: T,
      graph: &mut Graph,
    ) -> Vec<NodeIndex> {
      let mut little_nodes = vec![];
      for _ in 0..size {
        little_nodes.push(graph.add_op(op.clone()).finish())
      }
      little_nodes
    }

    fn connect_out_edges(x: NodeIndex, little_nodes: &Vec<NodeIndex>, graph: &mut Graph) {
      let out_edges: Vec<_> = graph
        .graph
        .edges_directed(x, Outgoing)
        .filter_map(|e| e.weight().as_data().map(|d| (d, e.target())))
        .collect();
      // assuming again that all outgoing shapes are the same
      // ^ NO, wrong, we dont assume that.
      for ((input_order, output_order, shape), target) in out_edges {
        // using output_order as the remembered index in logical shape
        // TODO: not recalculate the index_expressions so much
        let phys_index = match logical_to_physical(
          &(shape.index_expression(), shape.valid_expression()),
          output_order.into(),
        ) {
          Some(i) => i,
          None => {
            panic!("Something fucked up, outgoing edge index outside of expected physical size")
          }
        };
        info!(
          "phys={:?}, i={:?}, output={:?}",
          phys_index, input_order, output_order
        );
        graph.add_edge(
          little_nodes[phys_index],
          target,
          Dependency::Data {
            input_order,
            output_order: 0, // assuming single output
            shape: R0::to_tracker(),
          },
        );
      }
    }

    fn pointwise_op<T: Operator + 'static + Clone>(
      op: T,
      x: NodeIndex,
      size: usize,
      incoming: &Vec<((u8, u8, ShapeTracker), NodeIndex)>,
      graph: &mut Graph,
    ) -> Vec<NodeIndex> {
      let little_nodes = make_nodes(size, op, graph);
      connect_out_edges(x, &little_nodes, graph);

      for ((b, output_order, shape), source) in incoming {
        assert!(*output_order == 0, "Assuming sigle valued Op's");
        // assuming static shape
        let k = shape.n_elements().to_usize().unwrap();
        assert!( k == size, "Expected physical shape to be the same as incoming logical shape. size = {}, k = {}, src = {:?}", size, k, source ); // Op specific
        for j in 0..k {
          let (from, to) = (j as u8, j); // Op specific
          debug!("k={:?}, j={:?}, b={:?}", k, j, b);
          graph.add_edge(
            source.clone(),
            little_nodes[to],
            Dependency::Data {
              input_order: *b as u8,
              output_order: from, // saving the logical index that we used that edge for
              shape: *shape, // saving the original shape. TODO: save once, not in every little edge
            },
          );
        }
      }
      little_nodes
    }

    // precalculate all physical sizes as we're going to be removing edges
    let sizes = graph
      .node_identifiers()
      .map(|x| (x, get_own_size(x, graph)))
      .collect::<HashMap<_, _>>();
    info!("sizes: {:?}", sizes);

    let pi = {
      let mut pi = petgraph::algo::toposort(&graph.graph, None).unwrap();
      pi.reverse();
      pi
    };
    info!("first node in reverse toposorted graph: {:?}", pi[0]);

    // for every node:
    // 0. Match x on Op and arity
    // 1. Create pack of little nodes replacing x
    // 2. Connect outgoing edges, based on indices of the edges which from previous step are indexed like shape's logical indexes
    // 3. Create edges for incoming edges, connect as needed by the Op. Use output_order as indices, fix later when connecting from source.
    // 4. Remove x. Mark the new nodes for retrieval.
    for x in pi {
      info!("x={:?} in g={:?}", x, graph.graph);

      let incoming: Vec<_> = graph
        .edges_directed(x, Incoming)
        .filter_map(|e| e.weight().as_data().map(|d| (d, e.source())))
        .sorted_by_key(|((inp, _, _), _)| *inp)
        .collect();
      let size = sizes[&x];

      let little_nodes = if incoming.is_empty() {
        // TODO: treat Constants different to Input
        // x is source
        let little_nodes = make_nodes(size, InputOp {}, graph);
        connect_out_edges(x, &little_nodes, graph);
        little_nodes
      } else if let Some((((_, _, _), x),)) = incoming.iter().collect_tuple() {
        todo!("Unop")
      }
      // x is binop
      else if let Some((ll, rr)) = incoming.iter().collect_tuple() {
        if graph.check_node_type::<Add>(x) {
          debug!("Add {:?} {:?}", ll, rr);
          pointwise_op(Add {}, x, size, &incoming, graph)
        } else if graph.check_node_type::<Mul>(x) {
          debug!("Mul {:?} {:?}", ll, rr);
          pointwise_op(Mul {}, x, size, &incoming, graph)
        } else if graph.check_node_type::<LessThan>(x) {
          debug!("Mul {:?} {:?}", ll, rr);
          pointwise_op(LessThan {}, x, size, &incoming, graph)
        } else {
          todo!("Unsupported yet binop!") // are there any other binops we need? A: yes, comparisons
        }
      } else {
        // TODO: error handling
        panic!("unexpected node type")
      };

      // !!!
      mark_retrieve(&x, little_nodes, graph);
      graph.remove_node(x);
    }
  }
}

pub fn save_graphviz(path: String, graph: &Graph) -> Result<(), Box<dyn Error>> {
  use petgraph::dot::Dot;
  let dot = Dot::with_config(&graph.graph, &[]);
  let mut file = File::create(path)?;
  write!(file, "{:?}", dot)?;
  Ok(())
}

pub fn pretty_print_g(graph: &Graph) -> Result<(), Box<dyn Error>> {
  // TODO

  use petgraph_graphml::GraphMl;
  let a = GraphMl::new(&graph.graph).pretty_print(true);
  let mut str: Vec<u8> = vec![];
  let x = a.to_writer(&mut str)?;
  let str = String::from_utf8(str)?;
  // let str1 = str.as_ascii().into_iter().map(|x| x.clone()).collect::<Vec<_>>();
  println!("pretty g = {:?}", str);

  Ok(())
}

#[cfg(test)]
mod tests {
  use std::error::Error;

  use luminal::{
    graph::Graph,
    shape::{Const, R1, R2},
  };
  use tracing::info;

  use crate::{
    scalar::{pretty_print_g, save_graphviz},
    utils,
  };

  use super::ScalarCompiler;

  #[test]
  fn test_run() -> Result<(), Box<dyn Error>> {
    utils::init_logging()?;
    let mut cx = Graph::new();
    let a = cx.tensor::<R1<2>>().set(vec![1.0, 1.0]);
    let b = cx.tensor::<R1<2>>().set(vec![2.0, 2.0]);
    let d = cx.tensor::<R1<2>>().set(vec![3.0, 3.0]);
    let mut c = ((a + b) + d).retrieve();
    print!("{:?}", cx);
    save_graphviz("test_run_tensor.dot".to_string(), &cx)?;
    let r = cx.compile(ScalarCompiler::default(), &mut c);
    print!("{:?}", cx);
    print!("{:?}", r);
    // pretty_print_g(&cx)?;
    save_graphviz("test_run_scalar.dot".to_string(), &cx)?;
    cx.display();
    info!("compiled : {:?}", cx.graph);

    // THIS-WORKS
    // Open to see original graph  of (a+b)+d:
    // https://dreampuf.github.io/GraphvizOnline/#digraph%20%7B%0A%20%20%20%200%20%5B%20label%20%3D%20%22Tensor%20Load%22%20%5D%0A%20%20%20%201%20%5B%20label%20%3D%20%22Tensor%20Load%22%20%5D%0A%20%20%20%202%20%5B%20label%20%3D%20%22Tensor%20Load%22%20%5D%0A%20%20%20%203%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%204%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%200%20-%3E%203%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B2%5D%2C%20indexes%3A%20%5B0%5D%2C%20fake%3A%20%5Bf%0Aalse%5D%2C%20mask%3A%20%5B(0%2C%202147483647)%5D%2C%20padding%3A%20%5B(0%2C%200)%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%201%20-%3E%203%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B2%5D%2C%20indexes%3A%20%5B0%5D%2C%20fake%3A%20%5Bf%0Aalse%5D%2C%20mask%3A%20%5B(0%2C%202147483647)%5D%2C%20padding%3A%20%5B(0%2C%200)%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%203%20-%3E%204%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B2%5D%2C%20indexes%3A%20%5B0%5D%2C%20fake%3A%20%5Bf%0Aalse%5D%2C%20mask%3A%20%5B(0%2C%202147483647)%5D%2C%20padding%3A%20%5B(0%2C%200)%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%202%20-%3E%204%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B2%5D%2C%20indexes%3A%20%5B0%5D%2C%20fake%3A%20%5Bf%0Aalse%5D%2C%20mask%3A%20%5B(0%2C%202147483647)%5D%2C%20padding%3A%20%5B(0%2C%200)%5D%20%7D%20%7D%22%20%5D%0A%7D%0A
    // Open for scalar graph (see its basically double the original in this case):
    // https://dreampuf.github.io/GraphvizOnline/#digraph%20%7B%0A%20%20%20%200%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%201%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%203%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%204%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%205%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%206%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%207%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%208%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%209%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%2010%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%208%20-%3E%207%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%203%20-%3E%204%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%207%20-%3E%206%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%204%20-%3E%205%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%209%20-%3E%207%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%200%20-%3E%204%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%2010%20-%3E%206%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%0A%20mask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%201%20-%3E%205%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%7D%0A%0A

    Ok(())
  }

  #[test]
  fn test_run_2() -> Result<(), Box<dyn Error>> {
    utils::init_logging()?;
    let mut cx = Graph::new();
    let a = cx.tensor::<R1<2>>().set(vec![4.0, 4.0]);
    let b = cx.tensor::<R1<2>>().set(vec![8.0, 8.0]);
    let d = cx
      .tensor::<R2<2, 3>>()
      .set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    let mut c = ((a + b).expand::<(_, Const<3>), _>() + d).retrieve();
    print!("{:?}", cx);
    save_graphviz("test_run2_tensor.dot".to_string(), &cx)?;
    let r = cx.compile(ScalarCompiler::default(), &mut c);
    print!("{:?}", cx);
    print!("{:?}", r);
    // pretty_print_g(&cx)?;
    save_graphviz("test_run2_scalar.dot".to_string(), &cx)?;
    cx.display();
    info!("compiled : {:?}", cx.graph);

    // THIS-WORKS
    // Open to see original graph of (a+b).expand()+d:
    // https://dreampuf.github.io/GraphvizOnline/#digraph%20%7B%0A%20%20%20%200%20%5B%20label%20%3D%20%22Tensor%20Load%22%20%5D%0A%20%20%20%201%20%5B%20label%20%3D%20%22Tensor%20Load%22%20%5D%0A%20%20%20%202%20%5B%20label%20%3D%20%22Tensor%20Load%22%20%5D%0A%20%20%20%203%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%204%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%200%20-%3E%203%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B2%5D%2C%20indexes%3A%20%5B0%5D%2C%20fake%3A%20%5Bf%0Aalse%5D%2C%20mask%3A%20%5B(0%2C%202147483647)%5D%2C%20padding%3A%20%5B(0%2C%200)%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%201%20-%3E%203%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B2%5D%2C%20indexes%3A%20%5B0%5D%2C%20fake%3A%20%5Bf%0Aalse%5D%2C%20mask%3A%20%5B(0%2C%202147483647)%5D%2C%20padding%3A%20%5B(0%2C%200)%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%203%20-%3E%204%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B2%2C%203%5D%2C%20indexes%3A%20%5B0%2C%201%5D%2C%20fa%0Ake%3A%20%5Bfalse%2C%20true%5D%2C%20mask%3A%20%5B(0%2C%202147483647)%2C%20(0%2C%202147483647)%5D%2C%20padding%3A%20%5B(0%2C%200)%2C%20(0%2C%200)%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%202%20-%3E%204%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B2%2C%203%5D%2C%20indexes%3A%20%5B0%2C%201%5D%2C%20fa%0Ake%3A%20%5Bfalse%2C%20false%5D%2C%20mask%3A%20%5B(0%2C%202147483647)%2C%20(0%2C%202147483647)%5D%2C%20padding%3A%20%5B(0%2C%200)%2C%20(0%2C%200)%5D%20%7D%20%7D%22%20%5D%0A%7D%0A
    // Notice how it's the same graph as in test_run, but different shape at an edge
    // Open for scalar graph:
    // https://dreampuf.github.io/GraphvizOnline/#digraph%20%7B%0A%20%20%20%200%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%201%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%203%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%204%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%205%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%206%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%207%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%208%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%209%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%2010%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%2011%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%2012%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%2013%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%2014%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%2015%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%2016%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%2017%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%2018%20%5B%20label%20%3D%20%22Add%22%20%5D%0A%20%20%20%2012%20-%3E%2011%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%0A%2C%20mask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%203%20-%3E%204%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%2011%20-%3E%2010%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%0A%2C%20mask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%2011%20-%3E%209%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%0A%20mask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%2016%20-%3E%208%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%0A%20mask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%2015%20-%3E%207%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%0A%20mask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%2014%20-%3E%206%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%0A%20mask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%201%20-%3E%205%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%2011%20-%3E%208%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%0A%20mask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%204%20-%3E%207%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%204%20-%3E%206%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%204%20-%3E%205%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%200%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%2013%20-%3E%2011%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%0A%2C%20mask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%200%20-%3E%204%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%20%0Amask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%2018%20-%3E%2010%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%0A%2C%20mask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%20%20%20%2017%20-%3E%209%20%5B%20label%20%3D%20%22Data%20%7B%20input_order%3A%201%2C%20output_order%3A%200%2C%20shape%3A%20ShapeTracker%20%7B%20dims%3A%20%5B%5D%2C%20indexes%3A%20%5B%5D%2C%20fake%3A%20%5B%5D%2C%0A%20mask%3A%20%5B%5D%2C%20padding%3A%20%5B%5D%20%7D%20%7D%22%20%5D%0A%7D%0A

    Ok(())
  }
}

#[cfg(test)]
mod tests_other {
  use rand::{rngs::StdRng, SeedableRng};

  use luminal::prelude::*;

  use crate::scalar::ScalarCompiler;
  luminal::test_imports!();

  #[test]
  fn test_matmul() {
    let mut cx = Graph::new();
    let a = cx.tensor::<(Dyn<'M'>, Dyn<'K'>)>();
    let b = cx.tensor::<(Dyn<'K'>, Dyn<'N'>)>();
    let mut c = a.matmul(b).retrieve();

    cx.compile(ScalarCompiler::default(), &mut c);

    let d_dev = dfdx::prelude::Cpu::default();
    for m in (1..23).step_by(4) {
      for k in (1..35).step_by(3) {
        for n in (1..70).step_by(7) {
          let mut rng = StdRng::seed_from_u64(0);
          let a_data = random_vec_rng(m * k, &mut rng);
          let b_data = random_vec_rng(k * n, &mut rng);
          a.set_dyn(a_data.clone(), &[m, k]);
          b.set_dyn(b_data.clone(), &[k, n]);

          cx.execute();

          let d_a = d_dev.tensor_from_vec(a_data, (m, k));
          let d_b = d_dev.tensor_from_vec(b_data, (k, n));
          let d_c = d_a.matmul(d_b);

          assert_close_precision(&c.data(), &d_c.to_dtype::<f32>().as_vec(), 1e-2);
          c.drop();
        }
      }
    }
  }

  #[test]
  fn test_cpu_matmul_2d_2() {
    let mut cx = Graph::new();
    let a = cx.tensor::<R2<2, 3>>();
    a.set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    let b = cx.tensor::<R2<3, 4>>();
    b.set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    let mut c = a.matmul(b).retrieve();

    cx.execute();

    let unoptimized_c = c.data();
    cx.compile(ScalarCompiler::default(), &mut c);
    cx.execute();
    assert_close(&c.data(), &unoptimized_c);
  }
}

fn logical_to_physical((ind, val): &(BigExpression, BigExpression), index: usize) -> Option<usize> {
  if val.exec_single_var(index) != 0 {
    Some(ind.exec_single_var(index))
  } else {
    None
  }
}

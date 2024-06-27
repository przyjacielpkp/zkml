use luminal::graph::Graph;

// mod binary;

use std::{collections::HashMap, error::Error, fs::File, io::Write};

use tracing::{debug, info, instrument, warn};
use itertools::Itertools;
use petgraph::{visit::{EdgeRef, IntoNodeIdentifiers}, Direction::{Incoming, Outgoing}};

use luminal::{
    op::{Constant, ConstantValue, InputTensor, Operator},
    prelude::*,
    shape::Shape
};


/// 
/// Rewrite the ml source graph to scalarised version, where tensor operations are replaced with a multiplicity of scalar operations.
///  

/// Note: tensors may have dynamic dimensions.
/// Solution 1: break on these


// Asserts (in non-strictly-typed way) that all input tensors are single values.
pub struct ScalarGraph{ pub graph : Graph }


// Step1: after every node insert an aux node (at every out-edge). Make the node output vector of singleton tensors instead of single tensor of some length.
//       The step is unnessecary, its a mental simplification and helpful in debugging. 
// 
// Problem: What about nodes that output multiple values? Add, Mul, LessThan, ReduceAdd - are not like that right?

// Step2: Look at Operator taking as inputs aux nodes from step1 and rewrite it to a multitude of scalar ops.
// 

#[derive(Debug, Default)]
pub struct Aux {}

impl Operator for Aux {
  fn process(&mut self, _tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
    // semantically:
    // 
    // vec![tensors.into_iter().map(|(tensor, shape)| {
    //   assert!(shape.n_elements().to_usize().unwrap() == 1);
    //   tensor.only_element()
    // }).collect()];

    vec![]
  }
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
      let all_equal = graph.graph
        .edges_directed(node, Outgoing)
        .filter_map(|e| e.weight().as_data().map(|w| (w.1 /* hope equal 0 */, w.2)))
        .all_equal();
      assert!(all_equal, "All outgoing edges of a node must have the same shape.")
    }
  }
}

pub type ScalarCompiler = (
    // Step1,
    UniformOutShapes,
    Step2
);

#[derive(Debug, Default)]
pub struct Step1;

/// OBSOLETE
impl Compiler for Step1 {
  type Output = ();
  #[instrument(level = "debug", skip(_ids))]
  fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut _ids: T) {
    // Add Aux node at every outgoing edge in graph:
    //  - split nodes connected by edge
    //  - connect source to Aux with shape-many edges
    //  - connect Aux to target with 1 edge keeping original shape
    let xs = ((&graph.graph).node_indices().collect::<Vec<_>>()).clone();
    xs.into_iter().for_each(move |node| {
      let out_edges = graph.graph.edges_directed(node, Outgoing).map(|e| (e.weight().clone(), e.source(), e.target())).collect::<Vec<_>>();
      for (w, src, trg) in out_edges {
        if let Some((in_order, out_order, shape)) = w.as_data() {
          let n = shape.n_elements().to_usize().unwrap();
          let mut aux = graph.add_op(Aux{});
          for _ in 0..n {
            aux = aux.input(src, out_order,R0::to_tracker());
          }
          let aux_i = aux.finish();
          graph.add_edge(aux_i, trg, Dependency::Data { input_order: in_order, output_order: 0, shape: shape });
          let rem_edges = graph.edges_connecting(src, trg).map(|e| e.id()).collect::<Vec<_>>();
          rem_edges.into_iter().for_each(|e| {
            graph.remove_edge(e);
          });
        } else {
          continue;
        }
      }
    });
  }
}

#[derive(Debug, Default)]
pub struct Step2;

impl Compiler for Step2 {
    type Output = ();
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
      let mark_retrieve = |x: &NodeIndex, new_xs : Vec<_>, g: &mut Graph| {
        if let Some(w) = g.to_retrieve.get(x) {
          for new_x in new_xs {
            // let new_x : NodeIndex = new_x;
            g.to_retrieve.insert(new_x, (0 /* this probably refers to output index in Vec<Tensor> */, R0::to_tracker()));
          }
        }
      };

      let get_own_size = |x, gg: &Graph| {
        let get_own_shape = |x, gg: &Graph| {
          // reasonably we expect one of two cases: there is some outgoing edge OR it is a retrieval node
          if let Some(w) = gg.to_retrieve.get(&x) {
            w.clone().1
          } else {
            match gg.edges_directed(x.clone(),Outgoing).filter_map(|e| e.weight().as_data()).next() {
                Some((_,_,shape)) => {shape},
                None => { panic!("Add node has no outgoing edges and is not a retrieval node.") },
            }
          }
        };
        // assuming (and we have to) a staticly known shape
        match get_own_shape(x, gg).n_physical_elements().to_usize() {
          Some(n) => {n},
          None => { panic!("Node's output shape is not static.") },
        }
      };

      // precalculate all physical sizes as we're going to be removing edges
      let sizes = graph.node_identifiers()
        .map(|x| (x, get_own_size(x, graph)))
        .collect::<HashMap<_,_>>();
      info!("sizes: {:?}", sizes);

      let pi = {
        let mut pi = petgraph::algo::toposort(&graph.graph, None).unwrap();
        pi.reverse();
        pi
      };
      // error!("first node in reverse toposorted graph: {:?}", pi[0]);
      info!("first node in reverse toposorted graph: {:?}", pi[0]);
      // debug!("first node in reverse toposorted graph: {:?}", pi[0]); // todo: doenst work

      for x in pi {
        info!("x={:?} in g={:?}", x, graph.graph);
        // 0. Match x on Op and arity
        // 1. Create pack of little nodes replacing x
        // 2. Connect outgoing edges, based on indices of the edges which from previous step are indexed like shape's logical indexes
        // 3. Create edges for incoming edges, connect as needed by the Op. Use output_order as indices, fix later when connecting from source.
        // 4. Remove x. Mark the new nodes for retrieval.

        let incoming: Vec<_> = graph.edges_directed(x, Incoming)
          .filter_map(|e| e.weight().as_data().map(|d| (d, e.source())))
          .sorted_by_key(|((inp,_,_),_)| *inp )
          .collect();
        
        // x is source
        let little_nodes = if incoming.is_empty() {
          // split node into multiple nodes instead
          // for 
          let n = sizes[&x];
          let little_nodes = { 
              let mut little_nodes = vec![];
              for _ in 0..n { 
                little_nodes.push(graph.add_op(Add{}).finish())
              }
              little_nodes
            };
            
          let out_edges : Vec<_> = graph.graph.edges_directed(x, Outgoing)
              .filter_map(|e| e.weight().as_data().map(|d| (d, e.target()) ) )
              .collect()
              ;
          // out edges:
          for ((input_order, output_order, shape), target) in out_edges {
            // using output_order as the remembered index in logical shape
            // TODO: not recalculate the index_expressions so much 
            let phys_index = match logical_to_physical(&(shape.index_expression(),shape.valid_expression()), output_order.into()) {
                Some(i) => {i},
                None => { panic!("Something fucked up, outgoing edge index outside of expected physical size") },
            };
            graph.add_edge(
              little_nodes[ phys_index ] , 
              target,
              Dependency::Data{
                input_order, 
                output_order: 0, // assuming single output
                shape : R0::to_tracker()}
            );
          }
          // START HERE
          // need to remove x, below

          little_nodes
        }

        else if let Some((((_, _, _), x), )) = incoming.iter().collect_tuple() {
          todo!("Unop")
        }
        // x is binop
        else if let Some((ll, rr)) = incoming.into_iter().collect_tuple() {
          
          // x is Add
          if graph.check_node_type::<Add>(x) {

            info!("Add {:?} {:?}", ll, rr);

            let n = sizes[&x];
            let little_nodes = { 
              let mut little_nodes = vec![];
              for _ in 0..n { 
                little_nodes.push(graph.add_op(Add{}).finish())
              }
              little_nodes
            };

            let out_edges : Vec<_> = graph.graph.edges_directed(x, Outgoing)
              .filter_map(|e| e.weight().as_data().map(|d| (d, e.target()) ) )
              .collect()
              ;
            // assuming again that all outgoing shapes are the same
            // ^ NO, wrong, we dont assume that.

            // out edges:
            for ((input_order, output_order, shape), target) in out_edges {
              // using output_order as the remembered index in logical shape
              // TODO: not recalculate the index_expressions so much 
              let phys_index = match logical_to_physical(&(shape.index_expression(),shape.valid_expression()), output_order.into()) {
                  Some(i) => {i},
                  None => { panic!("Something fucked up, outgoing edge index outside of expected physical size") },
              };
              info!("n={:?}, phys={:?}, i={:?}, output={:?}", n, phys_index, input_order, output_order);
              graph.add_edge(
                little_nodes[ phys_index ] ,
                target,
                Dependency::Data{
                  input_order, 
                  output_order: 0, // assuming single output
                  shape : R0::to_tracker()}
              );
            }
            
            // edges l, r
            let in_edges = vec![ll, rr];
            for ((b, output_order, shape), source) in in_edges {
              assert!(output_order == 0, "Assuming sigle valued Op's");
              // assuming static shape
              let k = shape.n_elements().to_usize().unwrap();
              assert!( k == n, "In Add expected physical shape to same as incoming logical shape." ); // Op specific
              for j in 0..k {
                let (from, to) = (j as u8, j); // Op specific
                info!("n={:?}, k={:?}, j={:?}, b={:?}", n, k, j, b);
                graph.add_edge(
                  source.clone(),
                  little_nodes[to],
                  Dependency::Data{
                    input_order: b as u8,
                    output_order : from, // saving the logical index that we used that edge for
                    shape: R0::to_tracker(),
                  });
              }
            }
            little_nodes
          } else if graph.check_node_type::<Mul>(x) {
            todo!("Unsupoorted Mul");
          } else {
            todo!("Unsupported yet binop!")
          }
        // x is not that
        } else {
          // TODO: error handling
          panic!("unexpected node type")
        };

        // !!!
        mark_retrieve(&x, little_nodes, graph);
        graph.remove_node(x);
        
      }
        // x is Add
        // x is Mul
        // x is unop
        // x is source
      //   if graph.check_node_type::<Add>(x) {
      //     if 
      //   } else {

      //   }
      }
}

pub fn save_graphviz( path : String, graph : & Graph) -> Result<(), Box<dyn Error>> {
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
  let mut str: Vec<u8>  = vec![];
  let x = a.to_writer(&mut str)?;
  let str = String::from_utf8(str)?;
  // let str1 = str.as_ascii().into_iter().map(|x| x.clone()).collect::<Vec<_>>();
  println!("pretty g = {:?}", str);

  Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use luminal::{graph::Graph, shape::R1};
    use tracing::info;

    use crate::{scalar::{pretty_print_g, save_graphviz}, utils};

    use super::ScalarCompiler;

  #[test]
  fn test_run() -> Result<(), Box<dyn Error>> {
    utils::init_logging()?;
    let mut cx = Graph::new();
    let a = cx.tensor::<R1<2>>()
      .set(vec![1.0, 1.0]);
    let b = cx.tensor::<R1<2>>()
      .set(vec![2.0, 2.0]);
    let d = cx.tensor::<R1<2>>()
      .set(vec![3.0, 3.0]);
    let mut c = ((a + b) + d).retrieve();
    print!("{:?}", cx);
    let r = cx.compile(ScalarCompiler::default(), &mut c);
    print!("{:?}", cx);
    print!("{:?}", r);
    // pretty_print_g(&cx)?;
    save_graphviz("test_run.dot".to_string(), &cx)?;
    cx.display();
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

fn logical_to_physical(
  (ind, val): &(BigExpression, BigExpression),
  index: usize,
) -> Option<usize> {
  if val.exec_single_var(index) != 0 {
    Some( ind.exec_single_var(index) )
  } else {
    None
  }
}

pub(crate) fn constant(num: f32) -> SelectGraph {
  let mut n = op::<Constant>();
  n.check(move |o, _| {
      if let Some(c) = o.as_any().downcast_ref::<Constant>() {
          match c.0 {
              ConstantValue::Float(f) => f == num,
              _ => false,
          }
      } else {
          false
      }
  });
  n
}
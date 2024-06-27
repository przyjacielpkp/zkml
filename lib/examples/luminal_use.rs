use luminal::prelude::*;

use itertools::Itertools;
use std::collections::{HashMap, HashSet};

// use itertools::Itertools;
use petgraph::{
    algo::toposort,
    stable_graph::{NodeIndex, StableGraph},
    visit::EdgeRef,
    Direction,
};

use luminal::{
    op::{Add, Constant, ConstantValue, Function, MaxReduce, Mul, Operator, Recip, SumReduce},
    // prelude::*,
};

pub fn main() {
    // main2();
    main1()
}

pub fn main1() {

  // Setup graph and tensors (1)
  let mut cx = Graph::new();
  let a = cx.tensor::<R1<3>>()
      .set(vec![1.0, 2.0, 3.0]);
  let b = cx.tensor::<R1<3>>()
      .set(vec![1.0, 2.0, 3.0]);
  let d = cx.tensor::<R1<3>>()
    .set(vec![0.0, 1.0, 0.0]);

  // Actual operations (2)
  let c = ((a + b) + d).retrieve();
  // let xx: StableGraph<Box<dyn Operator>, Dependency> = cx.graph;
  println!("Graph: {:?}", cx);

  // let cx_below_infact_is : Graph = Graph { 
  // tensors: {},
  // dyn_map: {}, 
  // graph: StableGraph { Ty: "Directed", node_count: 5, edge_count: 4, edges: (0, 3), (1, 3), (3, 4), (2, 4), node weights: {
  //     0: Tensor Load, 
  //     1: Tensor Load,
  //     2: Tensor Load,
  //     3: Add, 
  //     4: Add
  //   }, 
  //   edge weights: {
  //     0: Data { input_order: 0, output_order: 0, shape: ShapeTracker { dims: [3], indexes: [0], fake: [false], mask: [(0, 
  //       2147483647)], padding: [(0, 0)] } },
  //     1: Data { input_order: 1, output_order: 0, shape: ShapeTracker { dims: [3], indexes: [0], fake: [false], mask: [(0, 
  //       2147483647)], padding: [(0, 0)] } }, 
  //     2: Data { input_order: 0, output_order: 0, shape: ShapeTracker { dims: [3], indexes: [0], fake: [false], mask: [(0, 
  //       2147483647)], padding: [(0, 0)] } }, 
  //     3: Data { input_order: 1, output_order: 0, shape: ShapeTracker { dims: [3], indexes: [0], fake: [false], mask: [(0,  
  //       2147483647)], padding: [(0, 0)] } }
  //   }, 
  //   free_node: NodeIndex(4294967295),
  //   free_edge: EdgeIndex(4294967295) },
  //   no_delete: {NodeIndex(4)}, 
  //   to_retrieve: {NodeIndex(4): (0, ShapeTracker { dims: [3], indexes: [0], fake: [false], mask: [(0, 2147483647)], 
  //     padding: [(0, 0)] })}, linearized_graph: None, consumers_map: None }
  //. 
  // where the Dependency::Data is Data { input_order: <index of tensor in the vector to operator>, output_order : <index of tensor in the previous output>, shape }
  // 
  // 
  // without the added 'd' tensor, the graph would be:
  // Graph { 
  //   tensors: {}, 
  //   dyn_map: {}, 
  //   graph: StableGraph { Ty: "Directed", node_count: 3, edge_count: 2, edges: (0, 2), (1, 2), node weights: {
  //     0: Tensor Load, 
  //     1: Tensor Load, 
  //     2: Add
  //   }, edge weights: {
  //     0: Data { input_order: 0, output_order: 0, shape: ShapeTracker { dims: [3], indexes: [0], fake:[false], mask: [(0, 2147483647)], padding: [(0, 0)] } }, 
  //     1: Data { input_order: 1, output_order: 0, shape: ShapeTracker { dims: [3], indexes: [0], fake: [false], mask: [(0, 2147483647)], padding: [(0, 0)] } }},
  //     free_node: NodeIndex(4294967295), 
  //     free_edge: EdgeIndex(4294967295) 
  //   }, 
  //   no_delete: {NodeIndex(2)},
  //   to_retrieve: {NodeIndex(2): (0, ShapeTracker { dims: [3], indexes: [0], fake: [false], mask: [(0, 2147483647)], padding: [(0, 0)] })}, 
  //   linearized_graph: None, consumers_map: None }
  
  // Run graph
  cx.execute();

  println!("Graph, executed: {:?}", cx);
  // Get result (4)
  println!("Result: {:?}", c);
  // Prints out [2.0, 4.0, 6.0]
}

pub fn main2() {

    // Setup graph and tensors (1)
    let mut cx = Graph::new();
    let a = cx.tensor::<R1<3>>()
        .set(vec![1.0, 2.0, 3.0]);
    let b = cx.tensor::<R2<3, 4>>()
        .set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0] );
    
    // Actual operations (2)
    // let c = (a + b).retrieve();
    // let xx: StableGraph<Box<dyn Operator>, Dependency> = cx.graph;
    // println!("Graph: {:?}", cx);
    
    // Run graph
    // cx.execute();
  
    // println!("Graph, executed: {:?}", cx);
    // println!("Result: {:?}", c);
  }
  

/// A simple linear layer
pub struct Linear<const A: usize, const B: usize> {
    pub(crate) weight: GraphTensor<R2<A, B>>,
}

impl<const A: usize, const B: usize> Module<GraphTensor<R1<A>>> for Linear<A, B> {
    type Output = GraphTensor<R1<B>>;

    fn forward(&self, input: GraphTensor<R1<A>>) -> Self::Output {
        input.matmul(self.weight)
    }
}


// thats how compilers are defined for luminal supported backends
// Q: do we follow the pattern?

/// Generic platform-agnostic optimizations. It's a good idea to use these all the time.
pub type GenericCompiler = (
    //RemoveSingleReductions,
    RemoveUnusedNodes,
    ArithmeticElimination,
    CSE,
);

/// [Common subexpression elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination)
#[derive(Default, Debug)]
pub struct CSE;

impl Compiler for CSE {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut ids: T) {
        // Look for nodes that have the exact same srcs
        // Loop cause I'm lazy
        let mut eliminated = true;
        while eliminated {
            eliminated = false;
            let mut srcs_set: HashMap<Vec<NodeIndex>, Vec<NodeIndex>> = HashMap::new();
            for node in graph.graph.node_indices().collect_vec() {
                if graph
                    .graph
                    .node_weight(node)
                    .unwrap()
                    .as_any()
                    .is::<Function>()
                {
                    continue;
                }
                let srcs = graph
                    .graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .filter(|e| !e.weight().is_schedule())
                    .sorted_by_key(|e| e.weight().as_data().unwrap().0)
                    .map(|e| e.source())
                    .collect_vec();

                if let Some(other_nodes) = srcs_set.get(&srcs) {
                    for other_node in other_nodes {
                        let a = graph.graph.node_weight(node).unwrap();
                        let Some(b) = graph.graph.node_weight(*other_node) else {
                            continue;
                        };
                        if format!("{a:?}") != format!("{b:?}") {
                            // Sloppy way to check if ops are equal, but we only expect primops here so it's ok
                            continue;
                        }
                        let a_src_shapes = graph
                            .get_sources(node)
                            .into_iter()
                            .map(|(_, _, a)| a)
                            .collect_vec();
                        let b_src_shapes = graph
                            .get_sources(*other_node)
                            .into_iter()
                            .map(|(_, _, a)| a)
                            .collect_vec();
                        if a_src_shapes != b_src_shapes {
                            continue;
                        }
                        // If the op, input shapes, and output shape is the same, we can combine them (UNCLEAR IF THIS IS TRUE, NEED PROPER PartialEq)
                        // Carry over outgoing edges from node to other_node
                        move_outgoing_edge(node, *other_node, &mut graph.graph);
                        // Transfer all references to node over to other node
                        remap(node, *other_node, &mut ids, graph);
                        // Remove node
                        graph.graph.remove_node(node);
                        eliminated = true;
                        break;
                    }
                    if eliminated {
                        break;
                    }
                }
                if let Some(nodes) = srcs_set.get_mut(&srcs) {
                    nodes.push(node);
                } else {
                    srcs_set.insert(srcs, vec![node]);
                }
            }
            srcs_set.clear();
        }
    }
}

/// Remove maxreduces and sumreduces that don't do anything
#[derive(Default)]
pub struct RemoveSingleReductions;

impl Compiler for RemoveSingleReductions {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut ids: T) {
        for node in graph.graph.node_indices().collect::<Vec<_>>() {
            let dim = if let Some(red) = graph
                .graph
                .node_weight(node)
                .unwrap()
                .as_any()
                .downcast_ref::<SumReduce>()
            {
                Some(red.0)
            } else {
                graph
                    .graph
                    .node_weight(node)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<MaxReduce>()
                    .map(|red| red.0)
            };
            if let Some(dim) = dim {
                if graph
                    .graph
                    .edges_directed(node, Direction::Incoming)
                    .next()
                    .map(|e| {
                        e.weight()
                            .as_data()
                            .map(|w| {
                                w.2.dims[w.2.indexes[dim]]
                                    .to_usize()
                                    .map(|i| i == 1)
                                    .unwrap_or_default()
                            })
                            .unwrap_or_default()
                    })
                    .unwrap_or_default()
                {
                    let upstream = graph
                        .graph
                        .neighbors_directed(node, Direction::Incoming)
                        .next()
                        .unwrap();
                    remap(node, upstream, &mut ids, graph);
                    move_outgoing_edge(node, upstream, &mut graph.graph);
                    graph.graph.remove_node(node);
                }
            }
        }
    }
}

/// Remove unused nodes
#[derive(Default, Debug)]
pub struct RemoveUnusedNodes;

impl Compiler for RemoveUnusedNodes {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _: T) {
        // Reverse topo sort
        for node in toposort(&graph.graph, None).unwrap().into_iter().rev() {
            if graph.edges_directed(node, Direction::Outgoing).count() == 0
                && !graph.no_delete.contains(&node)
            {
                // No dependencies and not marked for no_delete, so remove
                graph.remove_node(node);
            }
        }
    }
}

/// Enforce the graph gets ran in strictly depth-first order
#[derive(Default, Debug)]
pub struct DepthFirst;

impl Compiler for DepthFirst {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _: T) {
        fn toposort(
            id: NodeIndex,
            graph: &StableGraph<Box<dyn Operator>, Dependency>,
            visited: &mut HashSet<NodeIndex>,
        ) -> (Vec<NodeIndex>, usize, bool) {
            if visited.contains(&id) {
                return (vec![], 0, false);
            }
            // Loop through node sources
            let stacks = graph
                .edges_directed(id, Direction::Incoming)
                .sorted_by_key(|e| e.source())
                .map(|e| toposort(e.source(), graph, visited))
                .collect::<Vec<_>>();
            let num_stacks = stacks.len();

            let mut final_stack = vec![];
            let mut complete = true;
            for (mut stack, _, c) in stacks.into_iter().sorted_by_key(|(_, _, b)| !*b) {
                final_stack.append(&mut stack);
                complete &= c;
            }
            final_stack.push(id);
            visited.insert(id);

            (final_stack, num_stacks, complete)
        }

        // Depth-first toposort
        let mut visited = HashSet::default();
        let mut pre_sorted = petgraph::algo::toposort(&graph.graph, None).unwrap();
        pre_sorted.reverse();
        let mut stacks = vec![];
        for node in pre_sorted {
            if !visited.contains(&node) {
                stacks.push(toposort(node, &graph.graph, &mut visited));
            }
        }
        let mut nodes = vec![];
        for (mut stack, _, _) in stacks.into_iter().sorted_by_key(|(_, _, b)| !*b) {
            nodes.append(&mut stack);
        }

        // Insert schedule deps
        for i in 0..nodes.len() - 1 {
            graph.add_schedule_dependency(nodes[i], nodes[i + 1]);
        }
    }
}

/// **Reduces arithmetic expressions**
///
/// - Current: x + 0 => x, x * 1 => x
/// - TODO: x / x => 1, x - x => 0, x * 0 => 0, x - 0 => x, x * 0 => 0, 0 / x => 0
/// - TODO: Find a much cleaner way to do these eliminations
#[derive(Debug, Default)]
pub struct ArithmeticElimination;

impl Compiler for ArithmeticElimination {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut ids: T) {
        // x + 0, 0 + x
        let zero = constant(0.);
        let inp = node();
        let add1 = binary::<Add>(zero.clone(), inp.clone());
        let add2 = binary::<Add>(inp.clone(), zero.clone());
        let mut s1 = add1.clone().search(graph);
        let mut s2 = add2.clone().search(graph);
        while s1.next_match() || s2.next_match() {
            let (inp, zero, add) = if s1.matched {
                (s1.get(&inp), s1.get(&zero), s1.get(&add1))
            } else {
                (s2.get(&inp), s2.get(&zero), s2.get(&add2))
            };
            if graph.no_delete.contains(&zero) {
                continue;
            }
            // Carry over outgoing edges
            let input_shape = graph
                .graph
                .edges_connecting(inp, add)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            if input_shape.is_reshaped() {
                // If any output shape is non-contiguous, we need to keep the op for it's contiguous functionality TODO: replace with explicit contiguous op here
                if graph
                    .graph
                    .edges_connecting(inp, add)
                    .filter_map(|e| e.weight().as_data())
                    .any(|(_, _, sh)| sh.is_reshaped())
                {
                    continue;
                }
                for (weight, target) in graph
                    .graph
                    .edges_directed(add, petgraph::Direction::Outgoing)
                    .map(|e| (*e.weight(), e.target()))
                    .collect::<Vec<_>>()
                {
                    if let Some(weight) = weight.as_data() {
                        graph.graph.add_edge(
                            inp,
                            target,
                            Dependency::Data {
                                input_order: weight.0,
                                output_order: weight.1,
                                shape: input_shape,
                            },
                        );
                    }
                }
            } else {
                move_outgoing_edge(add, inp, &mut graph.graph);
            }
            remap(add, inp, &mut ids, graph);
            if graph
                .graph
                .edges_directed(zero, Direction::Outgoing)
                .count()
                == 1
            {
                graph.graph.remove_node(zero);
            }
            graph.graph.remove_node(add);
        }
        // x * 1, 1 * x
        let one = constant(1.);
        let inp = node();
        let mul1 = binary::<Mul>(one.clone(), inp.clone());
        let mul2 = binary::<Mul>(inp.clone(), one.clone());
        let mut s1 = mul1.clone().search(graph);
        let mut s2 = mul2.clone().search(graph);
        while s1.next_match() || s2.next_match() {
            let (inp, one, mul) = if s1.matched {
                (s1.get(&inp), s1.get(&one), s1.get(&mul1))
            } else {
                (s2.get(&inp), s2.get(&one), s2.get(&mul2))
            };
            if graph.no_delete.contains(&one) {
                continue;
            }
            // Carry over outgoing edges
            let input_shape = graph
                .graph
                .edges_connecting(inp, mul)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            if input_shape.is_reshaped() {
                // If any output shape is non-contiguous, we need to keep the op for it's contiguous functionality TODO: replace with explicit contiguous op here
                if graph
                    .graph
                    .edges_connecting(inp, mul)
                    .filter_map(|e| e.weight().as_data())
                    .any(|(_, _, sh)| sh.is_reshaped())
                {
                    continue;
                }
                for (weight, target) in graph
                    .graph
                    .edges_directed(mul, petgraph::Direction::Outgoing)
                    .map(|e| (*e.weight(), e.target()))
                    .collect::<Vec<_>>()
                {
                    if let Some(weight) = weight.as_data() {
                        graph.graph.add_edge(
                            inp,
                            target,
                            Dependency::Data {
                                input_order: weight.0,
                                output_order: weight.1,
                                shape: input_shape,
                            },
                        );
                    }
                }
            } else {
                move_outgoing_edge(mul, inp, &mut graph.graph);
            }
            remap(mul, inp, &mut ids, graph);
            graph.safe_remove_node(one, 1);
            graph.graph.remove_node(mul);
        }
        // recip(recip(x))
        let inp = node();
        let intermediate = unary::<Recip>(inp.clone());
        let out = unary::<Recip>(intermediate.clone());
        let mut s = out.clone().search(graph);
        while s.next_match() {
            let (inp, intermediate, out) = (s.get(&inp), s.get(&intermediate), s.get(&out));
            if graph.no_delete.contains(&intermediate) {
                continue;
            }
            // Carry over outgoing edges
            let input_shape = graph
                .graph
                .edges_connecting(inp, intermediate)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            if input_shape.is_reshaped() {
                // If any output shape is non-contiguous, we need to keep the op for it's contiguous functionality TODO: replace with explicit contiguous op here
                if graph
                    .graph
                    .edges_connecting(inp, intermediate)
                    .filter_map(|e| e.weight().as_data())
                    .any(|(_, _, sh)| sh.is_reshaped())
                    || graph
                        .graph
                        .edges_connecting(intermediate, out)
                        .filter_map(|e| e.weight().as_data())
                        .any(|(_, _, sh)| sh.is_reshaped())
                {
                    continue;
                }
                for (weight, target) in graph
                    .graph
                    .edges_directed(intermediate, petgraph::Direction::Outgoing)
                    .map(|e| (*e.weight(), e.target()))
                    .collect::<Vec<_>>()
                {
                    if let Some(weight) = weight.as_data() {
                        graph.graph.add_edge(
                            inp,
                            target,
                            Dependency::Data {
                                input_order: weight.0,
                                output_order: weight.1,
                                shape: input_shape,
                            },
                        );
                    }
                }
            } else {
                move_outgoing_edge(out, inp, &mut graph.graph);
            }
            remap(intermediate, inp, &mut ids, graph);
            remap(out, inp, &mut ids, graph);
            graph.remove_node(out);
            graph.safe_remove_node(intermediate, 0);
        }

        // exp2(log2(x))
        let inp = node();
        let intermediate = unary::<Exp2>(inp.clone());
        let out = unary::<Log2>(intermediate.clone());
        let mut s = out.clone().search(graph);
        while s.next_match() {
            let (inp, intermediate, out) = (s.get(&inp), s.get(&intermediate), s.get(&out));
            if graph.no_delete.contains(&intermediate) {
                continue;
            }
            // Carry over outgoing edges
            let input_shape = graph
                .graph
                .edges_connecting(inp, intermediate)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            if input_shape.is_reshaped() {
                // If any output shape is non-contiguous, we need to keep the op for it's contiguous functionality TODO: replace with explicit contiguous op here
                if graph
                    .graph
                    .edges_connecting(inp, intermediate)
                    .filter_map(|e| e.weight().as_data())
                    .any(|(_, _, sh)| sh.is_reshaped())
                    || graph
                        .graph
                        .edges_connecting(intermediate, out)
                        .filter_map(|e| e.weight().as_data())
                        .any(|(_, _, sh)| sh.is_reshaped())
                {
                    continue;
                }
                for (weight, target) in graph
                    .graph
                    .edges_directed(intermediate, petgraph::Direction::Outgoing)
                    .map(|e| (*e.weight(), e.target()))
                    .collect::<Vec<_>>()
                {
                    if let Some(weight) = weight.as_data() {
                        graph.graph.add_edge(
                            inp,
                            target,
                            Dependency::Data {
                                input_order: weight.0,
                                output_order: weight.1,
                                shape: input_shape,
                            },
                        );
                    }
                }
            } else {
                move_outgoing_edge(out, inp, &mut graph.graph);
            }
            remap(intermediate, inp, &mut ids, graph);
            remap(out, inp, &mut ids, graph);
            graph.remove_node(out);
            graph.safe_remove_node(intermediate, 0);
        }
    }
}

fn constant(num: f32) -> SelectGraph {
    let mut n = op::<Constant>();
    n.check(move |o, _| {
        if let Some(Constant(ConstantValue::Float(f), _)) = o.as_any().downcast_ref::<Constant>() {
            *f == num
        } else {
            false
        }
    });
    n
}
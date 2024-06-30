use luminal::{
  compiler_utils::{binary, node, Compiler, ToIdsMut},
  graph::Graph,
  op::Mul,
};
use tracing::instrument;

/// NOTE: Let's do that step already in snark compilation. Leaving the file for docs.
/// TODO: move the comments appropriately to snark module.

///
/// Defines a compilation step that changes the float computation to integer computation.
///
/// There's little to do:
///   We can take the float computation and just claim it's integer one - and that doesn't matter
///   until evaluation. As long as we are working on a static graph - the float computation looks same as integer one.
///   The only difference happens at multiplication (and at non-linear functions of sin/log etc. - though we cant* support these either way).
///
///

#[derive(Debug, Default)]
pub struct Integerize {
  scale: usize,
}

/// Kinda obsolete
impl Compiler for Integerize {
  type Output = ();
  #[instrument(level = "debug", skip(_ids))]
  ///
  /// Goes from float to integer computation.
  ///
  /// The computation graphs for ints and floats are simmilar.
  /// For multiplication we have to take care of the scaling.
  /// We'd like to just say: Mul_float a b => (Mul_int a' b') / scale
  /// But because can't divide (TODO: can we?) we instead take additional witness for the division result and say:
  ///   Mul_float a b => (if (Mul_int a' b' == witness * scale)) then witness else abort
  ///
  /// Also difference in evaluation of Op's: Functions and all the Sin/Log/Exp etc. In other words Add, AddReduce, MaxReduce - evaluate the same.
  /// But here that's not important.
  fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _ids: T) -> Self::Output {
    //
    // TODO: maybe snark frameworks already support that, so lets wait with implementation - it might be easier to do when producing snark.
    //
    // let l = node();
    // let r = node();
    // let mul = binary::<Mul>( l.clone(), r.clone() );
    // let mut s = mul.clone().search(graph);
    // while s.next_match() {
    //   let ll = s.get(&l);
    //   let rr = s.get(&r);
    //   let mm = s.get(&mul);

    //   // todo
    // }
  }
}

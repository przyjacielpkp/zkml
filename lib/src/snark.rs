use std::{collections::HashMap, fmt::Debug, hash::Hash};

use ark_ff::Field;
use ark_relations::{
  lc,
  r1cs::{ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError},
};
use itertools::Itertools;
///
/// Produce snark from the computation after scalar and integer transformations.
///
use luminal::{
  op::Add,
  prelude::{
    petgraph::{self, visit::EdgeRef, Direction::Incoming},
    NodeIndex,
  },
};
use tracing::{info, instrument};

use crate::scalar::{InputsTracker, ScalarGraph};

/// Tensor computation is initialized by setting input tensors data and then evaluating.
/// This function takes a mapping from input index to its tensor and creates a
/// mapping useful for the snark synthesis where input nodes are mapped to scalar values
/// using the given mapping from big tensor input nodes to little scalar input nodes.
pub fn snark_input_mapping<F: From<u128>>(
  assignment: HashMap<NodeIndex, Option<Vec<f32>>>,
  scale: usize,
  inputs_tracker: InputsTracker,
) -> HashMap<NodeIndex, SourceType<F>> {
  let mut result = HashMap::new();
  // privates
  assignment.into_iter().for_each(|(k, v)| match v {
    Some(vv) => inputs_tracker.new_inputs[&k]
      .iter()
      .zip(vv)
      .for_each(|(x, a)| {
        result.insert(x.clone(), SourceType::scaled_private(a, scale));
      }),
    None => inputs_tracker.new_inputs[&k].iter().for_each(|x| {
      result.insert(x.clone(), SourceType::Private(None));
    }),
  });
  // public
  inputs_tracker.constants.into_iter().for_each(|(k, v)| {
    result.insert(k, SourceType::scaled_public(v, scale));
  });
  result
}

#[derive(Debug)]
pub enum SourceType<F> {
  Private(Option<F>),
  Public(F),
}

impl<F: From<u128>> SourceType<F> {
  pub fn scaled_private(x: f32, scale: usize) -> Self {
    let y: u128 = (x * (scale as f32)).round() as u128;
    SourceType::Private(Some(F::from(y)))
  }
  pub fn scaled_public(x: f32, scale: usize) -> Self {
    let y: u128 = (x * (scale as f32)).round() as u128;
    SourceType::Public(F::from(y))
  }
}

#[derive(Debug)]
pub struct MLSnark {
  pub graph: ScalarGraph,
  pub scale: usize,
  pub private_inputs: HashMap<NodeIndex, Option<Vec<f32>>>,
}

/// NOTE on integer vs float computation: 
/// 
/// The ML computation is obviously meant to evaluate to floats.
/// If we were to take the static description of the expression for evaluation, but treat all Op's as if 
/// they act on integers - then what changes do we need to do to the expression? 
/// 
/// We define a scale factor and use integer `round(scale * f)` to represent a float `f`.
/// Firstly, we scale the inputs by scale factor.
/// Addition and  operations are fine as is.
/// Mul needs to divide the result by scale, sth along the lines for Recip, etc. LessThan probably needs to divide by scale (?).
/// In the end result is multiplied by scale.
/// 
/// Q: There is two ways in terms of code structure to implement this.
///    We can separate it into a compilation step or we can combine this step with snark synthesis.
/// Both are fine.
/// For example, in snark we see multiplication and
/// we'd like to just say: Mul_float a b => (Mul_int a' b') / scale
/// But because can't divide (TODO: can we?) we instead take additional witness for the division result and say:
///   Mul_float a b => (if (Mul_int a' b' == witness * scale)) then witness else abort
/// If doing a seperate integer step we'd say: Mul_float a b => (Div_int scale (Mul_int a' b'))
/// and then snark synthesis would rewrite Div_int to a similar circuit as above.
///

impl<F: Field> ConstraintSynthesizer<F> for MLSnark {
  // THIS-WORKS

  #[instrument(level = "debug", name = "generate_constraints")]
  fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
    let graph = &self.graph.graph;
    let source_map =
      snark_input_mapping(self.private_inputs, self.scale, self.graph.inputs_tracker);

    let pi = petgraph::algo::toposort(&graph.graph, None).unwrap();
    let mut vars: HashMap<NodeIndex, ark_relations::r1cs::Variable> = HashMap::new();
    // would actaully want:
    // let mut assignments: HashMap<NodeIndex, Box<dyn Fn()-> Result<F, SynthesisError>> > = HashMap::new();
    // but the below is easier to manage ownership with
    let mut assignments: HashMap<NodeIndex, Option<F>> = HashMap::new();
    // ^ thats silly that we need to track assignments but thats really because of the low level nature of arkworks api

    for x in pi {
      info!("x={:?}", x);

      let incoming: Vec<_> = graph
        .edges_directed(x, Incoming)
        .filter_map(|e| e.weight().as_data().map(|d| (d, e.source())))
        .sorted_by_key(|((inp, _, _), _)| *inp)
        .collect();

      // x is source
      if incoming.is_empty() {
        let src_ty = source_map
          .get(&x)
          .unwrap_or_else(|| panic!("Unknown source node {:?}!", x));
        use SourceType::*;
        let (v, ass) = match src_ty {
          Private(mn) => (
            cs.new_witness_variable(|| mn.ok_or(SynthesisError::AssignmentMissing))?,
            mn.clone(),
          ),
          Public(n) => (cs.new_input_variable(|| Ok(*n))?, Some(*n)),
        };
        vars.insert(x, v);
        assignments.insert(x, ass);
      } else if let Some((((_, _, _), x),)) = incoming.iter().collect_tuple() {
        todo!("Unop")
      }
      // x is binop
      else if let Some(((_, l), (_, r))) = incoming.into_iter().collect_tuple() {
        // assumes toposort order for unwraps
        let ll = vars.get(&l).unwrap().clone();
        let rr = vars.get(&r).unwrap().clone();
        let ll_val = assignments.get(&l).unwrap().clone();
        let rr_val = assignments.get(&r).unwrap().clone();

        let (v, ass) = if graph.check_node_type::<Add>(x) {
          // how nice would it be to do: (,) <*> ll_val <$> rr_val
          let ass = ll_val
            .and_then(|l| rr_val.map(|r| (l, r)))
            .map(|(l, r)| l + r);
          let v = cs.new_witness_variable(|| ass.ok_or(SynthesisError::AssignmentMissing))?;
          cs.enforce_constraint(
            lc!() + ll + rr,
            lc!() + ConstraintSystem::<F>::one(),
            lc!() + v,
          )?; // ll + rr s== v
          (v, ass)
        } else {
          todo!("Unsupported binops other than Add")
        };
        vars.insert(x, v);
        assignments.insert(x, ass);
      }
    }
    Ok(())
  }
}

#[cfg(test)]
mod test {
  // use super::*;
  // use ark_bls12_381::{Bls12_381, Fr as BlsFr};
  // use ark_groth16::Groth16;
  // use ark_snark::SNARK;
  //
  // #[test]
  // fn test_groth16_circuit_cubic() {
  //     let rng = &mut ark_std::test_rng();

  //     // generate the setup parameters
  //     let (pk, vk) = Groth16::<Bls12_381>::circuit_specific_setup(
  //         CubicDemoCircuit::<BlsFr> { x: None },
  //         rng,
  //     )
  //     .unwrap();

  //     // calculate the proof by passing witness variable value
  //     let proof1 = Groth16::<Bls12_381>::prove(
  //         &pk,
  //         CubicDemoCircuit::<BlsFr> {
  //             x: Some(BlsFr::from(3)),
  //         },
  //         rng,
  //     )
  //     .unwrap();

  //     // validate the proof
  //     assert!(Groth16::<Bls12_381>::verify(&vk, &[BlsFr::from(35)], &proof1).unwrap());

  //     // calculate the proof by passing witness variable value
  //     let proof2 = Groth16::<Bls12_381>::prove(
  //         &pk,
  //         CubicDemoCircuit::<BlsFr> {
  //             x: Some(BlsFr::from(4)),
  //         },
  //         rng,
  //     )
  //     .unwrap();
  //     assert!(Groth16::<Bls12_381>::verify(&vk, &[BlsFr::from(73)], &proof2).unwrap());

  //     assert!(!Groth16::<Bls12_381>::verify(&vk, &[BlsFr::from(35)], &proof2).unwrap());
  //     assert!(!Groth16::<Bls12_381>::verify(&vk, &[BlsFr::from(73)], &proof1).unwrap());
  // }
}

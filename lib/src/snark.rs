use std::{collections::HashMap, fmt::Debug, ops::Div};

use ark_bls12_381::Bls12_381;
use ark_bls12_381::Fr;
use ark_bls12_381::Fr as BlsFr;
use ark_ff::Field;
use ark_ff::Zero;
use ark_groth16::Groth16;
use ark_groth16::ProvingKey;
use ark_groth16::VerifyingKey;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::{boolean::Boolean, fields::fp::AllocatedFp, R1CSVar};
use ark_relations::{
  lc,
  r1cs::{ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError},
};
use ark_snark::SNARK;
use ark_std::cmp::Ordering::Less;
use itertools::Itertools;
// use ark_groth16::Groth16;
// use ark_snark::SNARK;

///
/// Produce snark from the computation after scalar and integer transformations.
///
use luminal::{
  op::{Add, LessThan, Mul, Recip},
  prelude::{
    petgraph::{self, visit::EdgeRef, Direction::Incoming},
    NodeIndex,
  },
};
use tracing::{info, instrument, warn};

use crate::model::copy_graph_roughly;
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

#[derive(Debug, Clone)]
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

pub type Curve = ark_bls12_381::Bls12_381;
pub type CircuitField = ark_bls12_381::Fr;

///
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
///  - Recip: n = f * s. 1/f = s/n. So we represent Recip(n) as s^2/n, where / is in F?
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
#[derive(Debug)]
pub struct MLSnark {
  pub graph: ScalarGraph,
  // start here
  pub scale: usize,
  // pub private_inputs: HashMap<NodeIndex, Option<Vec<f32>>>,
  pub source_map: HashMap<NodeIndex, SourceType<f32>>,
  // for convenience
  pub og_input_id: NodeIndex,
  // pub inputs_tracker : InputsTracker
}

pub type SourceMap = HashMap<NodeIndex, SourceType<f32>>;

impl MLSnark {
  pub fn set_input(&mut self, value: Vec<f32>) {
    set_input(
      &mut self.source_map,
      &self.graph.inputs_tracker,
      self.og_input_id,
      value,
    )
  }

  pub fn make_keys(
    &self,
  ) -> Result<(ProvingKey<Bls12_381>, VerifyingKey<Bls12_381>), SynthesisError> {
    let cloned = MLSnark {
      graph: self.graph.copy_graph_roughly(),
      scale: self.scale,
      source_map: self.source_map.clone(),
      og_input_id: self.og_input_id,
    };
    let rng = &mut ark_std::test_rng();
    // generate the setup parameters
    Groth16::<Bls12_381>::circuit_specific_setup(cloned, rng)
  }

  // pub fn make_proof
}

fn set_input(source_map: &mut SourceMap, tracker: &InputsTracker, id: NodeIndex, value: Vec<f32>) {
  let little_ids = tracker
    .new_inputs
    .get(&id)
    .unwrap_or_else(|| panic!("Wrong id"));
  for (little_id, v) in little_ids.into_iter().zip(value) {
    source_map.insert(*little_id, SourceType::Private(Some(v)));
  }
}

impl ConstraintSynthesizer<CircuitField> for MLSnark {
  // THIS-WORKS

  #[instrument(level = "debug", name = "generate_constraints")]
  fn generate_constraints(
    self,
    cs: ConstraintSystemRef<CircuitField>,
  ) -> Result<(), SynthesisError> {
    let graph = &self.graph.graph;
    let scale = self.scale;
    let scale_F = CircuitField::from(scale as u128);
    let source_map: HashMap<NodeIndex, SourceType<CircuitField>> = self
      .source_map
      .into_iter()
      .map(|(k, v)| {
        let v = match v {
          SourceType::Private(Some(x)) => SourceType::scaled_private(x, scale),
          SourceType::Private(None) => SourceType::Private(None),
          SourceType::Public(x) => SourceType::scaled_public(x, scale),
        };
        (k, v)
      })
      .collect();
    // snark_input_mapping(self.private_inputs, self.scale, self.graph.inputs_tracker);

    let pi = petgraph::algo::toposort(&graph.graph, None).unwrap();
    let mut vars: HashMap<NodeIndex, ark_relations::r1cs::Variable> = HashMap::new();
    // would actaully want:
    // let mut assignments: HashMap<NodeIndex, Box<dyn Fn()-> Result<F, SynthesisError>> > = HashMap::new();
    // but the below is easier to manage ownership with
    let mut assignments: HashMap<NodeIndex, Option<CircuitField>> = HashMap::new();
    // ^ thats silly that we need to track assignments but thats really because of the low level nature of arkworks api

    for x in pi {
      info!("x={:?}", x);

      let incoming: Vec<_> = graph
        .edges_directed(x, Incoming)
        .filter_map(|e| e.weight().as_data().map(|d| (d, e.source())))
        .sorted_by_key(|((inp, _, _), _)| *inp)
        .collect();

      let (v, ass) = {
        // SOURCE
        if incoming.is_empty() {
          let src_ty = source_map
            .get(&x)
            .unwrap_or_else(|| panic!("Unknown source node {:?}!", x));
          use SourceType::*;
          match src_ty {
            Private(mn) => (
              cs.new_witness_variable(|| mn.ok_or(SynthesisError::AssignmentMissing))?,
              mn.clone(),
            ),
            Public(n) => (cs.new_input_variable(|| Ok(*n))?, Some(*n)),
          }
        }
        // UNOP
        else if let Some((((_, _, _), y),)) = incoming.iter().collect_tuple() {
          let yy = vars.get(&y).unwrap().clone();
          let yy_val = assignments.get(&y).unwrap().clone();

          if graph.check_node_type::<Recip>(x) {
            // we have n = f * scale
            // The inverse is: 1/f = scale/n
            // so its represented by: m = scale * scale / n
            let ass = yy_val.map(|y| {
              scale_F.square()
                * y.inverse().unwrap_or_else(|| {
                  warn!("Tried inversing 0. Returning 0");
                  CircuitField::zero()
                })
            });
            let v = cs.new_witness_variable(|| ass.ok_or(SynthesisError::AssignmentMissing))?;
            cs.enforce_constraint(
              lc!() + yy,
              lc!() + v,
              lc!() + (scale_F * scale_F, ConstraintSystem::<CircuitField>::one()),
            )?; // m * n == scale * scale
            (v, ass)
          } else {
            todo!("Unsupported unop!")
          }
        }
        // BINOP
        else if let Some(((_, l), (_, r))) = incoming.into_iter().collect_tuple() {
          // assumes toposort order for unwraps
          let ll = vars.get(&l).unwrap().clone();
          let rr = vars.get(&r).unwrap().clone();
          let ll_val = assignments.get(&l).unwrap().clone();
          let rr_val = assignments.get(&r).unwrap().clone();

          if graph.check_node_type::<Add>(x) {
            // how nice would it be to do: (,) <*> ll_val <$> rr_val
            let ass = ll_val
              .and_then(|l| rr_val.map(|r| (l, r)))
              .map(|(l, r)| l + r);
            let v = cs.new_witness_variable(|| ass.ok_or(SynthesisError::AssignmentMissing))?;
            cs.enforce_constraint(
              lc!() + ll + rr,
              lc!() + ConstraintSystem::<CircuitField>::one(),
              lc!() + v,
            )?; // ll + rr == v
            (v, ass)
          } else if graph.check_node_type::<Mul>(x) {
            // ll * rr == tmp
            // v * scale == tmp
            let tmp_ass = ll_val
              .and_then(|l| rr_val.map(|r| (l, r)))
              .map(|(l, r)| l * r);
            let tmp =
              cs.new_witness_variable(|| tmp_ass.ok_or(SynthesisError::AssignmentMissing))?;
            let ass = tmp_ass.map(|x| x.div(scale_F));
            let v = cs.new_witness_variable(|| ass.ok_or(SynthesisError::AssignmentMissing))?;
            cs.enforce_constraint(lc!() + ll, lc!() + rr, lc!() + tmp)?;
            cs.enforce_constraint(
              lc!() + v,
              lc!() + (scale_F, ConstraintSystem::<CircuitField>::one()),
              lc!() + tmp,
            )?;
            (v, ass)
          } else if graph.check_node_type::<LessThan>(x) {
            // using the interface from r1cs_std here:
            let lll = FpVar::<Fr>::Var(AllocatedFp::new(ll_val, ll, cs.clone()));
            let rrr = FpVar::<Fr>::Var(AllocatedFp::new(ll_val, rr, cs.clone()));
            lll.enforce_cmp(&rrr, Less, false)?;

            let ass = || {
              lll.is_cmp(&rrr, Less, false).and_then(|is_cmp|
                // !!! so here remember that 1 is scale_F
                Boolean::<CircuitField>::le_bits_to_fp_var(&vec![is_cmp])?
                .value().map(|x| x * scale_F))
            };
            let ret = cs.new_witness_variable(ass)?;
            (ret, ass().ok())
          } else {
            panic!("Unsupported binop")
          }
        } else {
          panic!("No n-ary ops for n>2")
        }
      };
      vars.insert(x, v);
      assignments.insert(x, ass);
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

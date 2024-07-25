use std::convert::{TryFrom, TryInto};
use std::{collections::HashMap, fmt::Debug, ops::Div};

use ark_bls12_381::Bls12_381;
use ark_bls12_381::Fr;
use ark_ff::Zero;
use ark_ff::{BigInteger256, Field, PrimeField};
use ark_groth16::Groth16;
use ark_groth16::Proof;
use ark_groth16::ProvingKey;
use ark_groth16::VerifyingKey;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::fp::AllocatedFp;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::fields::FieldVar;
use ark_r1cs_std::uint128;
use ark_relations::{
  lc,
  r1cs::{ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError},
};
use ark_snark::SNARK;
use ark_std::cmp::Ordering::Less;
use blake2::digest::generic_array::typenum::uint;
use itertools::Itertools;

use luminal::prelude::petgraph::data::DataMap;
use luminal::prelude::petgraph::Direction::Outgoing;

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
use num_bigint::{BigInt, BigUint, ToBigInt};
use tracing::{info, instrument, warn};

use crate::scalar::ConstantOp;
use crate::scalar::InputOp;
// use crate::model::copy_graph_roughly;
use crate::scalar::{InputsTracker, ScalarGraph};
use crate::snark::scaling_helpers::*;

/// Tensor computation is initialized by setting input tensors data and then evaluating.
/// This function takes a mapping from input index to its tensor and creates a
/// mapping useful for the snark synthesis where input nodes are mapped to scalar values
/// using the given mapping from big tensor input nodes to little scalar input nodes.
pub fn snark_input_mapping(
  assignment: HashMap<NodeIndex, Option<Vec<f32>>>,
  scale: ScaleT,
  inputs_tracker: InputsTracker,
) -> HashMap<NodeIndex, SourceType<BigUint>> {
  let mut result = HashMap::new();
  // privates
  assignment.into_iter().for_each(|(k, v)| match v {
    Some(vv) => inputs_tracker.new_inputs[&k]
      .iter()
      .zip(vv)
      .for_each(|(x, a)| {
        result.insert(x.clone(), SourceType::scaled_private(a, &scale));
      }),
    None => inputs_tracker.new_inputs[&k].iter().for_each(|x| {
      result.insert(x.clone(), SourceType::Private(None));
    }),
  });
  result
}

#[derive(Debug, Clone)]
pub enum SourceType<F> {
  Private(Option<F>),
  Public(F),
}

impl SourceType<BigUint> {
  pub fn scaled_private(x: f32, scale: &ScaleT) -> Self {
    SourceType::Private(Some(positive_bigint(scaled_float(x, &scale))))
  }
  pub fn scaled_public(x: f32, scale: &ScaleT) -> Self {
    SourceType::Public(positive_bigint(scaled_float(x, &scale)))
  }
}

// ++
// A ++ B := A+B-z
// Operation that corresponds to the addition on floats being encoded by the uints by scaled_float mapping.
// See [Note: floats as ints]
fn add_add(a: BigInt, b: BigInt, scale: &ScaleT) -> BigInt {
  // todo: handle errors upstream
  a + b - scale.z
  // assert!( (a < u128::MAX - b) && ( a + b > z  ), "Addition doesn't overflow." );
  // r.try_into().ok()
}

#[derive(Debug, Clone)]
pub struct DivisionResult {
  result: BigInt,
  remainder: BigInt,
}

// **
// A ** B := (A * B + z*s + z*z - A*z - B*z) / s
// Operation that corresponds to the multiplication on floats being encoded by the uints by scaled_float mapping.
// See [Note: floats as ints]
pub fn mul_mul(a: BigInt, b: BigInt, scale: &ScaleT) -> DivisionResult {
  // todo: handle errors upstream
  let (s, z) = (scale.s, scale.z);
  let r = a.clone() * b.clone() + BigInt::from(z) * s + BigInt::from(z) * z - a * z - b * z;
  let (div, rem) = (r.clone() / BigInt::from(s), r.clone() % BigInt::from(s));
  DivisionResult {
    result: div,
    remainder: rem,
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
/// [Note: floats as integers]

/// ## intro
///
/// We map floats to a positive range of uints:
///
/// ```
/// [] : f32 -> uint
/// [x] = round(x*s) + z
/// ```
///
/// where s is the scale and z (positive) is the moved zero. Then the uints to the prime field elements.
///
/// ```
/// F : uint -> F
/// F(n) = 1 + 1 + ... + 1 (n times)
/// ```
///
/// We care to perform operations on such defined integers in a way that mimics the operations on floats.
/// This mapping has NO chance of being a homomorphism given the rounding.
/// Therefore, in the analysis we will be concerned instead with a subfield of floats, made only of floats being exactly represented with the set scale.
/// That is for floats where the above equation becomes:
///
/// ```
/// [x] = x*s + z
/// ```
///
/// and we assume that adding or multiplying such floats stays within the precision of the scale.
/// Is this really true about floats with all of their peculiarities?
/// I'm not certain, but the bottom line is that we round floats lossly to neighbouring floats and from now on calculations on these floats are homomorphic to the calculations we do with corresponsing uint's and field elements.
/// This means that the result of the calculation in the prime field almost matches the float result - it matches to a float thats close to the float that would be obtained had the calculation be performed on floats.
///
/// We will have:
///
/// ```
/// a + b ~ a' + b'
/// ```
///
/// where a, b are floats and a', b' are close floats rounded up to the scale precision,
/// and the homomorphic equality
///
/// ```
/// F[ a' + b' ] = F[a'] ++ F[b']
/// ```
///
/// where `++` is the corresponding addition (to be defined below) and the brackets around F's argument were omitted. Same for multiplication.
///
/// ### Overflow
///
/// Once i.e. two floats get added and the result overflows (or gets outside of some other range we've defined for ourselves),
/// we have these main options:
///
///  1. recognize the overflow and - trim the result OR overflow to match what floats do
///  2. error out in the proof production
///
/// Are there other options?
///
/// Option 1. has the problem that it needs to assert lessthan relation, which is done expensively via bit decomposition.
/// I'd choose option 2. because it is plausible to expect a trained neural net to not overflow. With proper care we might even cover the whole f32 range. Or we might not.
///
/// ## equations
///
/// Let's define the matching ++ and ** operations in the prime field. Let's take a, b (floats) already rounded to the scale precision. Let's ignore the overflows - that is i.e. when uint hits negative number we error out - and equations hold for non error results.
///
/// ```
/// F[ a + b ] = F( (a+b)*s+z ) = F( [a] + [b] - z ) = F[a] + F[b] - F(z)
/// ```
///
/// So we can define in F:
///
/// either option 1.:
///
/// ```
/// A ++ B := if (A+B)>F(z) {A+B-F(z)} else {F(0)}
/// ```
///
/// or error out:
///
/// ```
/// A ++ B := A+B-F(z)
/// ```
///
/// Multiplication:
///
/// ```
/// F[a] * F[b] = F([a] * [b]) = F( (a*s+z) * (b*s+z) ) = F( a*b*s*s + a*s*z + b*s*z + z*z ) = F((a*b*s*s) + z*s - z*s + (a*s+z)*z + (b*s+z)*z - z*z)
///
/// F[ a * b ] * F(s) := F( a*b*s*s + z*s ) = F( [a] * [b] + z*s + z*z - [a]*z - [b]*z )
/// that is:
/// F[ a * b ] * F(s) == F[a] * F[b] + F(z*s) + F(z*z) - F[a]*F(z) - F[b]*F(z)
/// ```
///
/// yielding
///
/// ```
/// A ** B == (A * B + F(z*s) + F(z*z) - A*F(z) - B*F(z)) / F(s)
/// ```
///
///
/// [Note: bigint]
///
/// We perform the calculations with bigints for simplicity and not to lose precision.
/// We recognize the overflow when mapping back into u128,
/// BUT still the operations done in the prime field may overflow, without us noticing leading to wrong results.
/// The solution is to add careful asserts on the ranges of subresults of all calculations - boring, lets do that later.
///
#[derive(Debug)]
pub struct MLSnark<F> {
  pub graph: ScalarGraph,
  // start here
  pub scale: ScaleT,
  // pub private_inputs: HashMap<NodeIndex, Option<Vec<f32>>>,
  pub source_map: HashMap<NodeIndex, SourceType<f32>>,
  // for convenience
  pub og_input_id: NodeIndex,
  // pub inputs_tracker : InputsTracker

  // this is needed due to some redundancy in how public inputs need to be passed to verify.
  // this field is filled up while calling SynthesizeSnark with assignments given to public inputs in order.
  // The few last elements record the result of the circuit, last element if single output. This is due to the topo ordering and model with single output vector, record more info if for our graph toposort stops guaranteeing that.
  // In practice: save this field after calling mk_proof. Share with the verifier.
  pub recorded_public_inputs: Vec<F>,
}

pub type SourceMap = HashMap<NodeIndex, SourceType<f32>>;

impl MLSnark<CircuitField> {
  /// Watch out: this needs to be called straight after make_proof.
  pub fn get_evaluation_result(&self) -> CircuitField {
    self.recorded_public_inputs.last().unwrap().clone()
  }

  pub fn set_input(&mut self, value: Vec<f32>) {
    set_input(
      &mut self.source_map,
      &self.graph.inputs_tracker,
      self.og_input_id,
      value,
    )
  }

  pub fn make_keys(
    &mut self,
  ) -> Result<(ProvingKey<Bls12_381>, VerifyingKey<Bls12_381>), SynthesisError> {
    let rng = &mut ark_std::test_rng();
    // generate the setup parameters
    Groth16::<Bls12_381>::circuit_specific_setup(self, rng)
  }

  // first provide all inputs with the set_input method, otherwise SynthesisError
  pub fn make_proof(
    &mut self,
    pk: &ProvingKey<Bls12_381>,
  ) -> Result<Proof<Bls12_381>, SynthesisError> {
    let rng = &mut ark_std::test_rng();
    Groth16::<Bls12_381>::prove(pk, self, rng)
  }
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

impl ConstraintSynthesizer<CircuitField> for &mut MLSnark<CircuitField> {
  /// Synthesize snark.
  ///
  /// We traverse the computation DAG in toposort order, assigning variables for a result of every node and asserting its value with snark constraints.
  /// We track the variable assignments (as bigints), which to us encode float values encoded as described in [Note: floats as ints].
  #[instrument(level = "debug", name = "generate_constraints-MLSnark")]
  fn generate_constraints(
    self,
    cs: ConstraintSystemRef<CircuitField>,
  ) -> Result<(), SynthesisError> {
    type F = CircuitField;
    let graph = &self.graph.graph;
    let scale = self.scale;
    let source_map: HashMap<NodeIndex, SourceType<BigUint>> = self
      .source_map
      .clone()
      .into_iter()
      .map(|(k, v)| {
        let v = match v {
          SourceType::Private(Some(x)) => SourceType::scaled_private(x, &scale),
          SourceType::Private(None) => SourceType::Private(None),
          SourceType::Public(x) => SourceType::scaled_public(x, &scale),
        };
        (k, v)
      })
      .collect();
    let mut public_record: Vec<F> = vec![];

    // return public input variable and assignment but also record it in the map
    let mk_public_input = |n: BigInt, public_record: &mut Vec<_>| {
      public_record.push(f_from_bigint_unsafe(n.clone()));
      let v = cs.new_input_variable(|| f_from_bigint(n.clone()))?;
      Ok((v, Some(n.clone())))
    };

    let pi = petgraph::algo::toposort(&graph.graph, None).unwrap();
    let mut vars: HashMap<NodeIndex, ark_relations::r1cs::Variable> = HashMap::new();
    let mut assignments: HashMap<NodeIndex, Option<BigInt>> = HashMap::new();

    for x in pi {
      let incoming: Vec<_> = graph
        .edges_directed(x, Incoming)
        .filter_map(|e| e.weight().as_data().map(|d| (d, e.source())))
        .sorted_by_key(|((inp, _, _), _)| *inp)
        .collect();

      let (v, ass) = {
        // SOURCE
        if incoming.is_empty() {
          if graph.check_node_type::<ConstantOp>(x) {
            let constant_op = graph
              .node_weight(x)
              .unwrap()
              .as_any()
              .downcast_ref::<ConstantOp>()
              .unwrap();
            let n = scaled_float(constant_op.val, &scale);
            mk_public_input(n, &mut public_record)?
          } else if graph.check_node_type::<InputOp>(x) {
            let src_ty = source_map
              .get(&x)
              .unwrap_or_else(|| panic!("Unknown source node {:?}!", x));
            use SourceType::*;
            match src_ty {
              Private(mn) => (
                cs.new_witness_variable(|| {
                  mn.clone()
                    .map(F::from)
                    .ok_or(SynthesisError::AssignmentMissing)
                })?,
                mn.clone().map(Into::into),
              ),
              Public(n) => mk_public_input(n.clone().into(), &mut public_record)?,
            }
          } else {
            panic!(
              "Unknown source type: {:?}",
              graph.node_weight(x).unwrap().type_name()
            )
          }
        }
        // UNOP
        else if let Some((((_, _, _), y),)) = incoming.iter().collect_tuple() {
          let yy = vars.get(&y).unwrap().clone();
          let yy_val = assignments.get(&y).unwrap().clone();

          if graph.check_node_type::<Recip>(x) {
            todo!("Fix recip to work with the changed float scaling.");
            // we have n = f * scale
            // The inverse is: 1/f = scale/n
            // so its represented by: m = scale * scale / n
            // let ass = yy_val.map(|y| {
            //   scale_f.square()
            //     * y.inverse().unwrap_or_else(|| {
            //       warn!("Tried inversing 0. Returning 0");
            //       CircuitField::zero()
            //     })
            // });
            // let v = cs.new_witness_variable(|| ass.ok_or(SynthesisError::AssignmentMissing))?;
            // cs.enforce_constraint(
            //   lc!() + yy,
            //   lc!() + v,
            //   lc!() + (scale_f * scale_f, ConstraintSystem::<CircuitField>::one()),
            // )?; // m * n == scale * scale
            // (v, ass)
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
            let ass = ll_val.zip_with(rr_val, |l, r| add_add(l, r, &scale));
            let v = cs.new_witness_variable(|| ass.clone().map(f_from_bigint).unwrap_or(Err(SynthesisError::AssignmentMissing)) /* would be cool to return two error types but in SyntheisError all other types are for internal use. bad design. */)?;
            // A ++ B := A+B-F(z)
            cs.enforce_constraint(
              lc!() + ll + rr,
              lc!() + ConstraintSystem::<CircuitField>::one(),
              lc!()
                + v
                + (
                  CircuitField::from(scale.z),
                  ConstraintSystem::<CircuitField>::one(),
                ),
            )?;
            (v, ass)
          } else if graph.check_node_type::<Mul>(x) {
            // A ** B := (A * B + F(z*s) + F(z*z) - A*F(z) - B*F(z)) / F(s)
            // that is, substituting some variables and rewriting division:
            // v * F(s) + Rem := l * r + F(z*s) + F(z*z) - l*F(z) - r*F(z)
            // v * F(s) + Rem := l * r + F(z*s) + F(z*z) - (l + r)*F(z)
            // v * F(s) + Rem := tmp1 + F(z*s) + F(z*z) - tmp2
            // getting constraints:
            // v * F(s) = tmp1 + F(z*s) + F(z*z) - tmp2 - Rem
            // l * r = tmp1
            // (l + r)*F(z) = tmp2
            let tmp1 = cs.new_witness_variable(|| {
              ll_val
                .clone()
                .zip_with(rr_val.clone(), |l, r| l * r)
                .and_then(|b| f_from_bigint(b).ok())
                .ok_or(SynthesisError::AssignmentMissing)
            })?;
            let tmp2 = cs.new_witness_variable(|| {
              ll_val
                .clone()
                .zip_with(rr_val.clone(), |l, r| {
                  f_from_bigint_unsafe((l + r) * scale.z)
                })
                .ok_or(SynthesisError::AssignmentMissing)
            })?;
            let div_res = ll_val
              .clone()
              .zip_with(rr_val.clone(), |l, r| mul_mul(l, r, &scale));
            let rem = cs.new_witness_variable(|| {
              div_res
                .clone()
                .ok_or(SynthesisError::AssignmentMissing)
                .and_then(|d| f_from_bigint(d.remainder))
            })?;
            let ass = div_res.map(|d| d.result);
            let v = cs.new_witness_variable(|| {
              ass
                .clone()
                .ok_or(SynthesisError::AssignmentMissing)
                .and_then(f_from_bigint)
            })?;
            let zs_zz = {
              let (s, z) = (BigInt::from(scale.s), BigInt::from(scale.z));
              (z.clone() * (s + z))
                .try_into()
                .map(|x: BigUint| F::from(x))
                .unwrap()
            };
            cs.enforce_constraint(
              lc!() + v,
              lc!() + (F::from(scale.s), ConstraintSystem::<CircuitField>::one()),
              lc!() + tmp1 - tmp2 - rem + (zs_zz, ConstraintSystem::<CircuitField>::one()),
            )?;
            cs.enforce_constraint(lc!() + ll, lc!() + rr, lc!() + tmp1)?;
            cs.enforce_constraint(
              lc!() + ll + rr,
              lc!() + (F::from(scale.z), ConstraintSystem::<CircuitField>::one()),
              lc!() + tmp2,
            )?;
            (v, ass)
          } else if graph.check_node_type::<LessThan>(x) {
            // witness assignments:
            //   x, y <- if l < r then (l, r) else (r, l)
            //   lt   <- (l < r)
            //
            // enforce:
            //    x = lt * l  + (1 - lt) * r          // x - r = lt * (l - r)
            //    y = lt * r  + (1 - lt) * l          // y - l = lt * (r - l)
            //    lt = 0 or 1                         // (lt - 1) * lt == 0
            //    x < y
            //
            // !!!
            // then lt_scaled = scaled_float(lt)

            let lr: Option<(_, _)> = ll_val.clone().and_then(|l| rr_val.clone().map(|r| (l, r)));
            let lt_ass_bool = lr.clone().map(|(l, r)| l < r);
            let lt_ass = lt_ass_bool
              .clone()
              .map(|b| if b { BigInt::from(1) } else { BigInt::from(0) });

            let make_xy = |noneg| {
              let ass = lt_ass_bool.clone().and_then(|b| {
                lr.clone()
                  .map(|(l, r)| if (if noneg { b } else { !b }) { l } else { r })
              }); // this can be written like above equation but is maybe faster
              Ok((
                cs.new_witness_variable(|| {
                  ass
                    .clone()
                    .ok_or(SynthesisError::AssignmentMissing)
                    .and_then(|x| f_from_bigint(x.clone()))
                })?,
                ass.clone(),
              ))
            };
            let (x, x_val) = make_xy(true)?;
            let (y, y_val) = make_xy(false)?;
            let lt = cs.new_witness_variable(|| {
              lt_ass
                .clone()
                .ok_or(SynthesisError::AssignmentMissing)
                .and_then(f_from_bigint)
            })?;

            cs.enforce_constraint(lc!() + lt, lc!() + ll - rr, lc!() + x - rr)?;
            cs.enforce_constraint(lc!() + lt, lc!() + rr - ll, lc!() + y - ll)?;
            cs.enforce_constraint(
              lc!() + lt,
              lc!() + ConstraintSystem::<CircuitField>::one() - lt,
              lc!(),
            )?;

            // using the interface from r1cs_std here:
            let xxx = FpVar::<Fr>::Var(AllocatedFp::new(
              x_val.map(|x| f_from_bigint(x.clone()).ok()).flatten(),
              x,
              cs.clone(),
            ));
            let yyy = FpVar::<Fr>::Var(AllocatedFp::new(
              y_val.map(|x| f_from_bigint(x.clone()).ok()).flatten(),
              y,
              cs.clone(),
            ));
            xxx.enforce_cmp(&yyy, Less, false)?;

            let lt_scaled_ass = lt_ass.clone().map(|lt| lt * scale.s + scale.z); // !! scaled_float
            let lt_scaled = cs.new_witness_variable(|| {
              lt_scaled_ass
                .clone()
                .ok_or(SynthesisError::AssignmentMissing)
                .and_then(f_from_bigint)
            })?;
            cs.enforce_constraint(
              lc!() + lt,
              lc!() + (F::from(scale.s), ConstraintSystem::<CircuitField>::one()),
              lc!() + lt_scaled - (F::from(scale.z), ConstraintSystem::<CircuitField>::one()),
            )?;

            // (lt_scaled, lt_scaled_ass)
            (lt_scaled, lt_scaled_ass)
          } else {
            panic!("Unsupported binop")
          }
        } else {
          panic!("No n-ary ops for n>2")
        }
      };

      vars.insert(x, v);
      assignments.insert(x, ass.clone());

      // if the node is a result node (a sink), assert its value against a public input.
      // we can do that only when creating the proof and having the private inputs,
      // so lets match on the Option. This all is quite a poor design but it follows from how arkworks is structured.
      if graph.edges_directed(x, Outgoing).next().is_none() {
        let z = cs.new_input_variable(|| {
          ass
            .clone()
            .ok_or(SynthesisError::AssignmentMissing)
            .and_then(f_from_bigint)
        })?;
        cs.enforce_constraint(
          lc!() + z,
          lc!() + ConstraintSystem::<CircuitField>::one(),
          lc!() + v,
        )?;
        match ass.clone() {
          Some(n) => public_record.push(f_from_bigint_unsafe(n)),
          None => {}
        }
      }
    }
    self.recorded_public_inputs = public_record;
    Ok(())
  }
}

mod tests {
  use num_bigint::BigInt;
  // use ark_ff::PrimeField;
  // use quickcheck::quickcheck;
  use super::bigints_close_as_floats;
  use crate::snark::scaling_helpers::*;
  use crate::snark::{mul_mul, CircuitField};
  use crate::SCALE;
  use proptest::num::f32::{NEGATIVE, POSITIVE};
  use proptest::prelude::*;
  use std::ops::Div;

  proptest! {

    #[test]
    fn test_scaling_is_mul_homo(a in -10e15..10e15f64, b in -10e15..10e15f64) {
      let scope = crate::utils::init_logging_tests();

      let a: f32 = a as f32;
      let b: f32 = b as f32;
      let a_m_b = a*b;
      // let scale_f: CircuitField = scaled_float(1.0, &SCALE);
      let a_m_b_f : BigInt = scaled_float(a_m_b, &SCALE);
      let a_f : BigInt = scaled_float(a, &SCALE);
      let b_f : BigInt = scaled_float(b, &SCALE);
      let af_m_bf = mul_mul(a_f.clone(), b_f.clone(), &SCALE).result;

      assert!(bigints_close_as_floats(af_m_bf, a_m_b_f, &SCALE), "scaled(a * b) == scaled(a) ** scaled(b)");
      drop(scope);
    }
  }
}

use std::{collections::HashMap, fmt::Debug, ops::Div};

use ark_bls12_381::Bls12_381;
use ark_bls12_381::Fr;
use ark_ff::Field;
use ark_ff::Zero;
use ark_groth16::Groth16;
use ark_groth16::Proof;
use ark_groth16::ProvingKey;
use ark_groth16::VerifyingKey;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::fields::fp::AllocatedFp;
use ark_relations::{
  lc,
  r1cs::{ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError},
};
use ark_snark::SNARK;
use itertools::Itertools;

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
use tracing::{instrument, warn};

use crate::scalar::ConstantOp;
use crate::scalar::InputOp;
// use crate::model::copy_graph_roughly;
use crate::scalar::{InputsTracker, ScalarGraph};
use crate::ScaleT;

/// Tensor computation is initialized by setting input tensors data and then evaluating.
/// This function takes a mapping from input index to its tensor and creates a
/// mapping useful for the snark synthesis where input nodes are mapped to scalar values
/// using the given mapping from big tensor input nodes to little scalar input nodes.
pub fn snark_input_mapping<F: From<i128>>(
  assignment: HashMap<NodeIndex, Option<Vec<f32>>>,
  scale: ScaleT,
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
  result
}

#[derive(Debug, Clone)]
pub enum SourceType<F> {
  Private(Option<F>),
  Public(F),
}

impl<F: From<i128>> SourceType<F> {
  pub fn scaled_private(x: f32, scale: ScaleT) -> Self {
    SourceType::Private(Some(scaled_float(x, scale)))
  }
  pub fn scaled_public(x: f32, scale: ScaleT) -> Self {
    SourceType::Public(scaled_float(x, scale))
  }
}

pub fn scaled_float<F: From<i128>>(x: f32, scale: ScaleT) -> F {
  // let y = (x. << f32::MANTISSA_DIGITS);
  let x : f64 = x.into();
  let s : f64 = scale as f64;
  let y: i128 = (x * s).round() as i128;
  F::from(y)

  // let x : f64 = x. .into();
  // let s : f64 = scale as f64;
  // let y: i128 = (x * s).round() as i128;
  // tracing::info!("scaled_float: x={:?}, s={:?}, y={:?}", x, s, y);
  // F::from(y)
}

// pub fn unscaled_field(f: CircuitField, scale: usize) -> f32 {
//   let x: u128 = f.try_into().unwrap();
//   let y: u128 = (x * (scale as f32)).round() as u128;
// }

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
    // let cloned = MLSnark {
    //   graph: self.graph.copy_graph_roughly(),
    //   scale: self.scale,
    //   source_map: self.source_map.clone(),
    //   og_input_id: self.og_input_id,
    // };
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

  #[instrument(level = "debug", name = "generate_constraints")]
  fn generate_constraints(
    self,
    cs: ConstraintSystemRef<CircuitField>,
  ) -> Result<(), SynthesisError> {
    let graph = &self.graph.graph;
    let scale = self.scale;
    let scale_f: CircuitField = scaled_float(1.0, scale);
    let source_map: HashMap<NodeIndex, SourceType<CircuitField>> = self
      .source_map
      .clone()
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
    let mut public_record = vec![];

    // return public input variable and assignment but also record it in the map
    let mk_public_input = |n, public_record: &mut Vec<_>| {
      public_record.push(n);
      let v = cs.new_input_variable(|| Ok(n))?;
      Ok((v, Some(n)))
    };

    let pi = petgraph::algo::toposort(&graph.graph, None).unwrap();
    let mut vars: HashMap<NodeIndex, ark_relations::r1cs::Variable> = HashMap::new();
    // would actaully want:
    // let mut assignments: HashMap<NodeIndex, Box<dyn Fn()-> Result<F, SynthesisError>> > = HashMap::new();
    // but the below is easier to manage ownership with
    let mut assignments: HashMap<NodeIndex, Option<CircuitField>> = HashMap::new();
    // ^ thats silly that we need to track assignments but thats really because of the low level nature of arkworks api

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
            let n = scaled_float(constant_op.val, scale);
            mk_public_input(n, &mut public_record)?
          } else if graph.check_node_type::<InputOp>(x) {
            let src_ty = source_map
              .get(&x)
              .unwrap_or_else(|| panic!("Unknown source node {:?}!", x));
            use SourceType::*;
            match src_ty {
              Private(mn) => (
                cs.new_witness_variable(|| mn.ok_or(SynthesisError::AssignmentMissing))?,
                mn.clone(),
              ),
              Public(n) => mk_public_input(*n, &mut public_record)?,
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
            // we have n = f * scale
            // The inverse is: 1/f = scale/n
            // so its represented by: m = scale * scale / n
            let ass = yy_val.map(|y| {
              scale_f.square()
                * y.inverse().unwrap_or_else(|| {
                  warn!("Tried inversing 0. Returning 0");
                  CircuitField::zero()
                })
            });
            let v = cs.new_witness_variable(|| ass.ok_or(SynthesisError::AssignmentMissing))?;
            cs.enforce_constraint(
              lc!() + yy,
              lc!() + v,
              lc!() + (scale_f * scale_f, ConstraintSystem::<CircuitField>::one()),
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
            let ass = tmp_ass.map(|x| x.div(scale_f));
            let v = cs.new_witness_variable(|| ass.ok_or(SynthesisError::AssignmentMissing))?;
            cs.enforce_constraint(lc!() + ll, lc!() + rr, lc!() + tmp)?;
            cs.enforce_constraint(
              lc!() + v,
              lc!() + (scale_f, ConstraintSystem::<CircuitField>::one()),
              lc!() + tmp,
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
            // then lt_scaled = lt * scale_F

            let lr: Option<(_, _)> = ll_val
              .and_then(|l| rr_val.map(|r| (l, r)));
            let lt_ass_bool = lr.map(|(l, r)|  l < r );
            let lt_ass = lt_ass_bool.map(|b| if b {CircuitField::from(1 as i64)} else {CircuitField::zero()});

            let make_xy = |noneg| {
              let ass = lt_ass_bool.and_then(|b| lr.map(|(l, r)|
                if (if noneg {b} else {! b}) {l} else {r}));  // this can be written like above equation but is maybe faster
              Ok((cs.new_witness_variable(|| ass.ok_or(SynthesisError::AssignmentMissing))?, ass))
            };
            let (x, x_val) = make_xy(true)?;
            let (y, y_val) = make_xy(false)?;
            let lt = cs.new_witness_variable(|| lt_ass.ok_or(SynthesisError::AssignmentMissing))?;

            cs.enforce_constraint(
              lc!() + lt,
              lc!() + ll - rr,
              lc!() + x - rr
            )?;
            cs.enforce_constraint(
              lc!() + lt,
              lc!() + rr - ll,
              lc!() + y - ll
            )?;
            // todo: uncomment
            // cs.enforce_constraint(
            //   lc!() + lt,
            //   lc!() + ConstraintSystem::<CircuitField>::one() - lt,
            //   lc!() + ConstraintSystem::<CircuitField>::zero()
            // )?;

            // using the interface from r1cs_std here:
            let xxx = FpVar::<Fr>::Var(AllocatedFp::new(x_val, x, cs.clone()));
            let yyy = FpVar::<Fr>::Var(AllocatedFp::new(y_val, y, cs.clone()));
            // xxx.enforce_cmp(&yyy, Less, false)?;
            
            let lt_scaled_ass = lt_ass.map(|lt| lt * scale_f);
            let lt_scaled = cs.new_witness_variable(|| lt_scaled_ass.ok_or(SynthesisError::AssignmentMissing))?;
            cs.enforce_constraint(
              lc!() + lt,
              lc!() + (scale_f, ConstraintSystem::<CircuitField>::one()),
              lc!() + lt_scaled,
            )?;

            (lt_scaled, lt_scaled_ass)
          } else {
            panic!("Unsupported binop")
          }
        } else {
          panic!("No n-ary ops for n>2")
        }
      };
      vars.insert(x, v);
      assignments.insert(x, ass);
      tracing::info!("{:?}: {:?} = {:?}", x, v, ass);

      // if the node is a result node (a sink), assert its value against a public input.
      // we can do that only when creating the proof and having the private inputs,
      // so lets match on the Option. This all is quite a poor design but it follows from how arkworks is structured.
      if graph.edges_directed(x, Outgoing).next().is_none() {
        let z = cs.new_input_variable(|| ass.ok_or(SynthesisError::AssignmentMissing))?;
        cs.enforce_constraint(
          lc!() + z,
          lc!() + ConstraintSystem::<CircuitField>::one(),
          lc!() + v,
        )?;
        match ass {
          Some(n) => public_record.push(n),
          None => {}
        }
      }
    }
    self.recorded_public_inputs = public_record;
    Ok(())
  }
}

pub fn field_elems_close(a : CircuitField , b : CircuitField, scale: ScaleT) -> bool {
  (a - b).square().le(
    & ( (a.square() + b.square()) * scaled_float::<CircuitField>(0.01, scale) ) )  
}

mod tests {
    // use ark_ff::PrimeField;
    // use quickcheck::quickcheck;
    use proptest::prelude::*;
    use proptest::num::f32::{POSITIVE, NEGATIVE};
    use std::ops::Div;
    use crate::snark::{field_elems_close, scaled_float, CircuitField};
    use crate::SCALE;

  proptest! {

    #[test]
    fn test_scaling_is_mul_homo(a in -10e15..10e15f64, b in -10e15..10e15f64) {
      let scope = crate::utils::init_logging_tests();
      let a: f32 = a as f32;
      let b: f32 = b as f32;
      let scale = SCALE;
      let a_m_b = a*b;
      let scale_f: CircuitField = scaled_float(1.0, scale);
      let a_m_b_f : CircuitField = scaled_float(a_m_b, scale);
      let a_f : CircuitField = scaled_float(a, scale);
      let b_f : CircuitField = scaled_float(b, scale);
      let a_f_m_b_f = a_f * b_f;
      let a_f_m_b_f_d_s: CircuitField = (a_f * b_f).div(scale_f);
      let diff = field_elems_close(a_f_m_b_f_d_s , a_m_b_f, scale);
      tracing::info!("bool={:?}, a*b={:?}, a_f={:?}, b_f={:?}, a_f*b_f={:?}, (a_f*b_f)/s_f={:?}, (a*b)_f={:?}", 
        diff, a_m_b, a_f, b_f, a_f_m_b_f, a_f_m_b_f_d_s, a_m_b_f);
      prop_assert!(diff, "scaled(a * b) == scaled(a) * scaled(b) / scale");
      drop(scope);
    }
  }

}
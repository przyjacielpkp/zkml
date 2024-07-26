use std::convert::{TryFrom, TryInto};
use std::{fmt::Debug, ops::Div};

use super::CircuitField;
use ark_ff::PrimeField;
use ark_relations::r1cs::SynthesisError;
use num_bigint::{BigInt, BigUint, ToBigInt};

#[derive(Debug, Clone, Copy)]
/// Defines a scaling of a float by: x => round(s * x) + z
pub struct ScaleT {
  pub s: u128,
  pub z: u128,
}

/// Convert a float to a scaled integer.
///
/// See [Note: floats as ints]
pub fn scaled_float(x: f32, scale: &ScaleT) -> BigInt {
  // // TODO: handle errors upstream
  let s = scale.s;
  let z = scale.z;
  let x: f64 = x.into();
  // assert!( (- (z as f64) / (s as f64) <= x) && (x <= (z as f64) / (s as f64))  , "Float within allowed range");
  let scaled: BigInt = ((x * (s as f64)).round())
    .to_bigint()
    .expect("scaled_float: Conversion to bigint failed");
  scaled + z
  // todo: handle the unwrap upstream
  // assert!(y.is_positive(), "Scaled float outside of the range!");
  // assert!( ((((y - z) / s) as f64) - x).abs() <= x * 0.0001  , "Float is recoverable");
}

// TODO: factor out a module with conversion helpers
pub fn unscaled_f(x: CircuitField, scale: &ScaleT) -> Option<f32> {
  unscaled_bigint(f_to_bigint(x), scale)
}

pub fn unscaled_bigint(x: BigInt, scale: &ScaleT) -> Option<f32> {
  // // TODO: handle errors upstream
  let s = scale.s;
  let z = scale.z;
  let div: i128 = ((x.clone() - z) / s).try_into().ok()?;
  let rem: i128 = ((x - z) % s).try_into().ok()?;
  Some(((div as f64) + ((rem as f64) / (s as f64))) as f32)
  // todo: handle the unwrap upstream
  // assert!(y.is_positive(), "Scaled float outside of the range!");
  // assert!( ((((y - z) / s) as f64) - x).abs() <= x * 0.0001  , "Float is recoverable");
}

pub fn positive_bigint(b: BigInt) -> BigUint {
  b.try_into()
    .expect("Expects positive bigint, otherwise its negative float overflow")
}
pub fn f_from_bigint(b: BigInt) -> Result<CircuitField, SynthesisError> {
  CircuitField::try_from(positive_bigint(b)).map_err(|_| SynthesisError::AssignmentMissing)
}
pub fn f_from_bigint_unsafe(b: BigInt) -> CircuitField {
  f_from_bigint(b)
    .expect("Expects bigint to fit in the prime field range, otherwise its positive float overflow")
}

pub fn f_to_bigint(a: CircuitField) -> BigInt {
  let x: BigUint = a.into_repr().into();
  x.into()
}

pub fn field_elems_close(a: CircuitField, b: CircuitField, scale: ScaleT) -> bool {
  let a = f_to_bigint(a);
  let b = f_to_bigint(b);
  let diff = if a < b {
    b.clone() - a.clone()
  } else {
    a.clone() - b.clone()
  };
  diff.le(&((a.max(b)).div(scale.s * 100)))
}

pub fn floats_close(a: f32, b: f32) -> bool {
  let prec = 0.01; // could be better ;/ todo: improve division accuracy by returning the remainder
  (a - b).abs().le(&(prec * (a.abs() + b.abs()).max(1.0)).max(
    1.0 / 100_000.0, /* todo: should directly depend on scale */
  ))
}
pub fn bigints_close_as_floats(a: BigInt, b: BigInt, scale: &ScaleT) -> bool {
  let ab = || {
    let aa = unscaled_bigint(a, scale)?;
    let bb = unscaled_bigint(b, scale)?;
    Some((aa, bb))
  };
  match ab() {
    None => false,
    Some((a, b)) => floats_close(a, b),
  }
}

pub fn field_close_as_floats(a: CircuitField, b: CircuitField, scale: &ScaleT) -> bool {
  let ab = || {
    let aa = unscaled_f(a, scale)?;
    let bb = unscaled_f(b, scale)?;
    Some((aa, bb))
  };
  match ab() {
    None => false,
    Some((a, b)) => floats_close(a, b),
  }
}

pub mod scaling_helpers;
mod snark;

pub use snark::*;

use ark_bls12_381::{Bls12_381, FrParameters};
use ark_ff::Fp256;

pub type Pairing = Bls12_381;
pub type Fp = Fp256<FrParameters>;

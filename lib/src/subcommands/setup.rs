use std::{
  collections::HashMap,
  fs::File,
  path::{Path, PathBuf},
};

use ark_groth16::Groth16;
use ark_serialize::CanonicalSerialize;
use ark_snark::SNARK;
use luminal::compiler_utils::ToId;
use rand::{rngs::StdRng, SeedableRng};

use crate::snark::Curve;

pub struct Setup {
  dataset_path: PathBuf,
  prover_output_path: PathBuf,
  verifier_output_path: PathBuf,
  weights_output_path: PathBuf,
}

impl Setup {
  pub fn new(
    dataset_path: &Path,
    prover_output_path: &Path,
    verifier_output_path: &Path,
    weights_output_path: &Path,
  ) -> Self {
    Self {
      dataset_path: PathBuf::from(dataset_path),
      prover_output_path: PathBuf::from(prover_output_path),
      verifier_output_path: PathBuf::from(verifier_output_path),
      weights_output_path: PathBuf::from(weights_output_path),
    }
  }

  pub fn run(self) {
    let rng = StdRng::seed_from_u64(1);

    let dataset = crate::model::read_dataset(self.dataset_path.as_path());
    let (graph, model) = crate::model::run_model(dataset);

    let weights: HashMap<_, _> = crate::model::get_weights(&graph, &model)
      .iter()
      .map(|(key, val)| (crate::utils::unpack_node_index(key.clone()), val.clone()))
      .collect();

    // TODO: replace ... with proper object, so that weights will be treated as the public input,
    // then, the following lines can be uncommented
    /*let circuit = crate::lib::compile(...);

    let (pk, vk) = Groth16::<Curve>::circuit_specific_setup(circuit, &mut rng).unwrap();

    crate::utils::canonical_serialize_to_file(&self.prover_output_path, &pk);
    crate::utils::canonical_serialize_to_file(&self.verifier_output_path, &vk);*/
    crate::utils::serialize_to_file(&self.weights_output_path, &weights);
  }
}

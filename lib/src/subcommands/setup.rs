use std::path::{Path, PathBuf};

use ark_groth16::Groth16;
use ark_snark::SNARK;
use rand::{rngs::StdRng, SeedableRng};

use crate::{model::TrainingParams, snark::Curve};

pub struct Setup {
  training_params: TrainingParams,
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
    epochs: usize,
  ) -> Self {
    Self {
      training_params: TrainingParams {
        data: crate::model::read_dataset(dataset_path),
        epochs,
      },
      prover_output_path: PathBuf::from(prover_output_path),
      verifier_output_path: PathBuf::from(verifier_output_path),
      weights_output_path: PathBuf::from(weights_output_path),
    }
  }

  pub fn run(self) {
    let mut rng = StdRng::seed_from_u64(1);

    let trained_graph = crate::model::run_model(self.training_params);
    let weights: Vec<_> = trained_graph
      .weights
      .iter()
      .map(|(key, val)| (crate::utils::unpack_node_index(*key), val.clone()))
      .collect();

    let circuit = crate::compile(trained_graph);
    let (pk, vk) = Groth16::<Curve>::circuit_specific_setup(circuit, &mut rng).unwrap();

    crate::utils::canonical_serialize_to_file(&self.prover_output_path, &pk);
    crate::utils::canonical_serialize_to_file(&self.verifier_output_path, &vk);
    crate::utils::serialize_to_file(&self.weights_output_path, &weights);
  }
}

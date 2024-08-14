use ark_serialize::CanonicalSerialize;

use crate::model::TrainingParams;
use std::path::{Path, PathBuf};

pub struct Setup {
  prover_output_path: PathBuf,
  verifier_output_path: PathBuf,
  weights_output_path: PathBuf,
  public_inputs_output_path: PathBuf,
  training_params: TrainingParams,
}

impl Setup {
  pub fn new(
    dataset_path: &Path,
    prover_output_path: &Path,
    verifier_output_path: &Path,
    weights_output_path: &Path,
    public_inputs_output_path: &Path,
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
      public_inputs_output_path: PathBuf::from(public_inputs_output_path),
    }
  }

  pub fn run(self) {
    let trained_model = crate::model::run_model(self.training_params);

    let mut snark = crate::compile(&trained_model.graph);
    let (pk, vk) = snark.make_keys().unwrap();

    let weights: Vec<(u32, Vec<f32>)> = trained_model
      .graph
      .weights
      .iter()
      .map(|(key, val)| (crate::utils::unpack_node_index(key), val.clone()))
      .collect();

    let public_inputs: Vec<_> = snark
      .recorded_public_inputs
      .iter()
      .map(|val| {
        let mut buff: Vec<u8> = vec![];
        val.serialize(&mut buff).unwrap();
        buff
      })
      .collect();

    crate::utils::canonical_serialize_to_file(&self.prover_output_path, &pk);
    crate::utils::canonical_serialize_to_file(&self.verifier_output_path, &vk);
    crate::utils::serialize_to_file(&self.weights_output_path, &weights);
    crate::utils::serialize_to_file(&self.public_inputs_output_path, &public_inputs);
  }
}

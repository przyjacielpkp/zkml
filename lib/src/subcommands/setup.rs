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

use crate::{model::TrainParams, snark::Curve};

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
    epichs: usize,
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

    let dataset = crate::model::read_dataset(self.dataset_path.as_path()).unwrap();
    let graph = crate::model::run_model(TrainParams {
      data: dataset,
      epochs: 20,
    });
    // todo: implement serialization for TrainedGraph, then recreate test_trained_into_snark.

    // let weights = crate::model::get_weights(&graph, &model);

    // we need to construct input to compile function
    // let circuit = crate::lib::compile(...);

    // let (pk, vk) = Groth16::<Curve>::circuit_specific_setup(circuit, &mut rng).unwrap();

    // let mut pk_file =
    //   File::create(self.prover_output_path).expect("Failed to create prover setup file");
    // pk.serialize_uncompressed(pk_file);

    // let mut vk_file =
    //   File::create(self.verifier_output_path).expect("Failed to create verifier setup file");
    // vk.serialize_uncompressed(vk_file);

    // let mut weights_file =
    //   File::create(self.weights_output_path).expect("Failed to create weights file");
    // vk.serialize_uncompressed(weights_file);
  }
}

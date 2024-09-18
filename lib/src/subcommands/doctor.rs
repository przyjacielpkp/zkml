use std::path::Path;

use ark_groth16::{Proof, ProvingKey};

use crate::{
  model::GraphForSnark,
  snark::Pairing,
  utils::{canonical_deserialize_from_file, deserialize_from_file},
};

pub struct Doctor {
  url: String,
  pk: ProvingKey<Pairing>,
  trained_model: GraphForSnark,
}

impl Doctor {
  pub fn new(pk_input_path: &Path, model_input_path: &Path, url: String) -> Self {
    let pk = canonical_deserialize_from_file(pk_input_path);
    let weights: Vec<(u32, Vec<f32>)> = deserialize_from_file(model_input_path);
    let trained_model = crate::model::from_weights(weights);
    Self {
      url,
      pk,
      trained_model,
    }
  }

  pub async fn run(self) {
    let client = reqwest::Client::new();

    let mut snark = crate::compile(&self.trained_model);
    snark.set_input(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let proof: Proof<Pairing> = snark.make_proof(&self.pk).expect("Cannot read proving key");

    let mut doctored_public_inputs = snark.recorded_public_inputs;
    doctored_public_inputs[9] = crate::snark::CircuitField::from(0);
    let body = super::packet::pack(proof, doctored_public_inputs);
    let response = match client.get(self.url).body(body).send().await {
      Ok(response) => response,
      Err(err) => panic!("{}", err),
    };
    let res = response.text().await.unwrap();
    println!("{}", res);
  }
}

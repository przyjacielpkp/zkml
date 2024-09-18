use std::path::Path;

use ark_groth16::{Proof, ProvingKey};

use crate::{
  model::GraphForSnark,
  snark::Pairing,
  utils::{canonical_deserialize_from_file, deserialize_from_file},
};

pub struct Client {
  url: String,
  pk: ProvingKey<Pairing>,
  input: Vec<f32>,
  trained_model: GraphForSnark,
}

impl Client {
  pub fn new(
    input_path: &Path,
    pk_input_path: &Path,
    model_input_path: &Path,
    url: String,
  ) -> Self {
    let pk = canonical_deserialize_from_file(pk_input_path);
    let input = deserialize_from_file(input_path);
    let weights: Vec<(u32, Vec<f32>)> = deserialize_from_file(model_input_path);
    let trained_model = crate::model::from_weights(weights);
    Self {
      url,
      pk,
      input,
      trained_model,
    }
  }

  pub async fn run(self) {
    let client = reqwest::Client::new();

    let mut snark = crate::compile(&self.trained_model);
    snark.set_input(self.input);
    let proof: Proof<Pairing> = snark.make_proof(&self.pk).expect("Cannot read proving key");

    let body = super::packet::pack(proof, snark.recorded_public_inputs);
    let response = match client.get(self.url).body(body).send().await {
      Ok(response) => response,
      Err(err) => panic!("{:?}", err.is_connect()),
    };
    let res = response.text().await.unwrap();
    println!("{}", res);
  }
}

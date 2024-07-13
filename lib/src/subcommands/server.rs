use std::{collections::HashMap, path::Path};

use ark_groth16::r1cs_to_qap::LibsnarkReduction;
use ark_groth16::{Groth16, VerifyingKey};
use ark_snark::SNARK;
use axum::{routing::get, Router};
use luminal::prelude::NodeIndex;

#[derive()]
pub struct Server {
  port: u16,
  verifying_key: VerifyingKey<crate::snark::Curve>,
  weights: HashMap<NodeIndex, Vec<f32>>,
}

impl Server {
  pub fn new(port: u16, weights_path: &Path, verifying_key_path: &Path) -> Self {
    let verifying_key = crate::utils::canonical_deserialize_from_file(verifying_key_path);

    let weights = crate::utils::deserialize_from_file::<HashMap<u32, Vec<f32>>>(weights_path)
      .iter()
      .map(|(key, val)| (NodeIndex::from(key.clone()), val.clone()))
      .collect();

    Self {
      port,
      verifying_key,
      weights,
    }
  }

  pub async fn run(self) {
    tracing_subscriber::fmt::init();

    let server_addr = format!("0.0.0.0:{}", self.port);

    let app = Router::new().route("/", get(Self::handle_request));
    let tcp_listener = tokio::net::TcpListener::bind(server_addr).await.unwrap();
    axum::serve(tcp_listener, app).await.unwrap();
  }

  async fn handle_request(input: String) -> String {
    let verified = Groth16::<_>::verify(&self.verifying_key, &self.weights, &proof).unwrap();
    serde_json::to_string(&verified).unwrap()
  }
}

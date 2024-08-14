use crate::snark::{Fp, Pairing};
use ark_groth16::{Groth16, Proof, VerifyingKey};
use ark_serialize::CanonicalDeserialize;
use ark_snark::SNARK;
use axum::{extract::State, routing::get, Router};
use std::{path::Path, sync::Arc};

struct ServerState {
  vk: VerifyingKey<Pairing>,
  public_input: Vec<Fp>,
}

pub struct Server {
  port: u16,
  vk: VerifyingKey<Pairing>,
  public_input: Vec<Fp>,
}

impl Server {
  pub fn new(port: u16, vk_input_path: &Path, public_input_path: &Path) -> Self {
    Self {
      port,
      vk: crate::utils::canonical_deserialize_from_file(vk_input_path),
      public_input: crate::utils::deserialize_from_file::<Vec<Vec<u8>>>(public_input_path)
        .iter()
        .map(|str| Fp::deserialize(str.as_slice()).unwrap())
        .collect(),
    }
  }

  pub async fn run(self) {
    tracing_subscriber::fmt::init();

    let server_addr = format!("0.0.0.0:{}", self.port);

    let state = Arc::<ServerState>::new(ServerState {
      vk: self.vk,
      public_input: self.public_input,
    });
    let app = Router::new()
      .route("/", get(Self::handle_request))
      .with_state(state);
    let tcp_listener = tokio::net::TcpListener::bind(server_addr).await.unwrap();
    axum::serve(tcp_listener, app).await.unwrap();
  }

  async fn handle_request(State(state): State<Arc<ServerState>>, input: String) -> String {
    let proof = Proof::<Pairing>::deserialize(input.as_bytes());
    let result = match proof {
      Ok(proof) => Groth16::<Pairing>::verify(&state.vk, state.public_input.as_slice(), &proof)
        .unwrap_or(false),
      Err(_) => false,
    };
    serde_json::to_string(&result).unwrap()
  }
}

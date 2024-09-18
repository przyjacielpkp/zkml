use ark_groth16::{Groth16, VerifyingKey};
use ark_snark::SNARK;
use axum::{extract::State, routing::get, Router};
use std::{path::Path, sync::Arc};

use crate::snark::Pairing;

struct ServerState {
  vk: VerifyingKey<Pairing>,
}

pub struct Server {
  port: u16,
  vk: VerifyingKey<Pairing>,
}

impl Server {
  pub fn new(port: u16, vk_input_path: &Path) -> Self {
    Self {
      port,
      vk: crate::utils::canonical_deserialize_from_file(vk_input_path),
    }
  }

  pub async fn run(self) {
    let server_addr = format!("0.0.0.0:{}", self.port);

    let state = Arc::<ServerState>::new(ServerState { vk: self.vk });
    let app = Router::new()
      .route("/", get(Self::handle_request))
      .with_state(state);
    let tcp_listener = tokio::net::TcpListener::bind(server_addr).await.unwrap();
    axum::serve(tcp_listener, app).await.unwrap();
  }

  async fn handle_request(State(state): State<Arc<ServerState>>, input: String) -> String {
    println!("Got request: {}", input);
    let result = match super::packet::unpack(&input) {
      Ok((proof, public_input)) => {
        Groth16::<Pairing>::verify(&state.vk, public_input.as_slice(), &proof).unwrap()
      }
      Err(_) => false,
    };
    serde_json::to_string(&result).unwrap()
  }
}

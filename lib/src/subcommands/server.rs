use axum::{routing::get, Router};

pub struct Server {
  port: u16,
}

impl Server {
  #[allow(clippy::new_without_default)]
  pub fn new(port: u16) -> Self {
    Self { port }
  }

  pub async fn run(self) {
    tracing_subscriber::fmt::init();

    let server_addr = format!("0.0.0.0:{}", self.port);

    let app = Router::new().route("/", get(Self::handle_request));
    let tcp_listener = tokio::net::TcpListener::bind(server_addr).await.unwrap();
    axum::serve(tcp_listener, app).await.unwrap();
  }

  async fn handle_request(input: String) -> String {
    let verified = true;
    /*
     *let proof = parse(input);
     *let verified = verify(proof);
     */
    serde_json::to_string(&verified).unwrap()
  }
}

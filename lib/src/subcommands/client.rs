use std::{
  fs,
  path::{Path, PathBuf},
};

pub struct Client {
  input: String,
  url: String,
}

impl Client {
  pub fn new(input_path_buf: PathBuf, url: String) -> Self {
    let input = match fs::read_to_string(input_path_buf.clone()) {
      Ok(val) => val,
      Err(_) => panic!(
        "Failed to open the file: {}",
        input_path_buf.to_str().unwrap()
      ),
    };
    Self { input, url }
  }

  pub async fn run(self) {
    let client = reqwest::Client::new();

    let proof = "";
    /*
     *let proof = generate_proof(self.input);
     */
    let response = match client.get(self.url).body(proof).send().await {
      Ok(response) => response,
      Err(err) => panic!("{}", err),
    };
    let res = response.text().await.unwrap();
    println!("{}", res);
  }
}

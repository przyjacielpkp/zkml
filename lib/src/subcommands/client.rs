use std::path::PathBuf;

pub struct Client {
  input_file: PathBuf,
  url: String,
}

impl Client {
  pub fn new(input_file: PathBuf, url: String) -> Self {
    Self { input_file, url }
  }

  pub fn run(self) {
    println!(
      "Client run with args: input_file = {}, url = {}",
      self.input_file.to_str().unwrap(),
      self.url
    );
    todo!("Client implementation");
  }
}

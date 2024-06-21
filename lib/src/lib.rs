use tracing::info;


#[derive(Debug)]
pub struct CompileConfig {
  // whatever
  pub artifacts: bool,
}

#[derive(Debug)]
pub enum Error {

}

#[tracing::instrument]
pub fn main(config: CompileConfig) -> Result<(), Error> {
  info!("Entering main");
  Ok(())
} 
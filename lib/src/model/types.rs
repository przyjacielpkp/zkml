pub const INPUT_DIMENSION: usize = 9;

pub type InputsVec = Vec<[f32; INPUT_DIMENSION]>;
pub type OutputsVec = Vec<f32>;

pub type Dataset = (Vec<[f32; INPUT_DIMENSION]>, Vec<f32>);

pub struct TrainingParams {
  pub data: (InputsVec, OutputsVec),
  pub epochs: usize,
}

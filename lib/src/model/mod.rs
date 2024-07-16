// todo: abstract away the training loop. split from the lib crate

pub mod lessthan_model;
pub mod medium_model;
pub mod tiny_model;

pub use medium_model::*;


use serde::Deserialize;

/// [Partially defined] AppConfig - fields only to provide example.
/// Also defines the config file format (Option fields can be omitted).
#[derive(Debug, Deserialize)]
pub struct AppConfig {
  /// Defines some setting
  pub some_option: Option<bool>,
  /// Whether to save intermediate representations
  pub artifacts: Option<bool>
}

impl AppConfig {
  // merge configs where the second overwrites the first
  pub fn merge(self, other: Self) -> Self {
    Self {
      artifacts: other.artifacts.or(self.artifacts),
      some_option: other.some_option.or(self.some_option),
    }
  }
}

impl Default for AppConfig {
  fn default() -> Self {
    Self {
      artifacts: None,
      some_option: None,
    }
  }
}

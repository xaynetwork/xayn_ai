pub mod bert;
pub mod classifier;
pub mod cnn;

use std::io::Error as IoError;

use displaydoc::Display;
use thiserror::Error;
use tract_onnx::prelude::TractError;

/// The potential errors of the models.
#[derive(Debug, Display, Error)]
pub enum ModelError {
    /// Failed to read the onnx model: {0}
    Read(#[from] IoError),
    /// Failed to run a tract operation: {0}
    Tract(#[from] TractError),
    /// Invalid onnx model shapes
    Shape,
}

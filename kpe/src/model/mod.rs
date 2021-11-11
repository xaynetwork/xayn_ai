pub mod bert;
pub mod classifier;
pub mod cnn;

use std::io::Error as IoError;

use displaydoc::Display;
use ndarray::ShapeError;
use thiserror::Error;
use tract_onnx::prelude::TractError;

use layer::{conv::ConvError, io::LoadingLayerFailed};

/// The potential errors of the models.
#[derive(Debug, Display, Error)]
pub enum ModelError {
    /// Failed to read the onnx model: {0}
    Read(#[from] IoError),

    /// Failed to run a tract operation: {0}
    Tract(#[from] TractError),

    /// Invalid array shapes: {0}
    Shape(#[from] ShapeError),

    /// Failed to read or run the CNN model: {0}
    Cnn(#[from] ConvError),

    /// Failed to read the Classifier model: {0}
    Classifier(#[from] LoadingLayerFailed),

    /// Remaining parameters must be used: {0:?}
    UnusedParams(Vec<String>),
}

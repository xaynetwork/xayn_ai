use std::{
    io::{Error as IoError, Read},
    sync::Arc,
};

use derive_more::{Deref, From};
use displaydoc::Display;
use thiserror::Error;
use tract_onnx::prelude::{
    tvec,
    Datum,
    Framework,
    InferenceFact,
    InferenceModelExt,
    Tensor,
    TractError,
    TypedModel,
    TypedSimplePlan,
};

use crate::{
    ndarray::{Dim, Dimension, Ix3},
    tokenizer::Encodings,
};

/// A wrapped onnx model.
pub struct Model {
    plan: TypedSimplePlan<TypedModel>,
    shape: Ix3,
}

/// The potential errors of the model.
#[derive(Debug, Display, Error)]
pub enum ModelError {
    /// Failed to read the onnx model: {0}
    Read(#[from] IoError),
    /// Failed to run a tract operation: {0}
    Tract(#[from] TractError),
    /// Invalid onnx model shapes
    Shape,
}

/// The predicted encodings.
#[derive(Clone, Deref, From)]
pub struct Predictions(pub Arc<Tensor>);

impl Model {
    /// Creates a model from an onnx model file.
    ///
    /// Requires the batch and token size of the model inputs.
    pub fn new(
        // `Read` instead of `AsRef<Path>` is needed for wasm
        mut model: impl Read,
        batch_size: usize,
        token_size: usize,
    ) -> Result<Self, ModelError> {
        let input_fact = InferenceFact::dt_shape(i64::datum_type(), &[batch_size, token_size]);
        let plan = tract_onnx::onnx()
            .model_for_read(&mut model)?
            .with_input_fact(0, input_fact.clone())?
            .with_input_fact(1, input_fact.clone())?
            .with_input_fact(2, input_fact)?
            .into_optimized()?
            .into_runnable()?;

        let shape = plan
            .model()
            .output_fact(0)?
            .shape
            .as_finite()
            .map(|os| os.get(0..3).map(|os| Dim([os[0], os[1], os[2]])))
            .flatten()
            .ok_or(ModelError::Shape)?;
        // input/output shapes are guaranteed to match when a sound onnx model is loaded
        debug_assert_eq!(&[batch_size, token_size], &shape.slice()[0..2]);

        Ok(Model { plan, shape })
    }

    /// Runs prediction on encoded sequences.
    pub fn predict(&self, encodings: Encodings) -> Result<Predictions, ModelError> {
        let inputs = tvec!(
            encodings.token_ids.0.into(),
            encodings.attention_masks.0.into(),
            encodings.type_ids.0.into()
        );
        let mut outputs = self.plan.run(inputs)?;

        Ok(outputs.remove(0).into())
    }

    /// Returns the batch size of the model.
    pub fn batch_size(&self) -> usize {
        self.shape[0]
    }

    /// Returns the token size of the model.
    pub fn token_size(&self) -> usize {
        self.shape[1]
    }

    /// Returns the embedding size of the model.
    pub fn embedding_size(&self) -> usize {
        self.shape[2]
    }
}

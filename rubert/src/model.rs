use std::io::{Error as IoError, Read};

use derive_more::{Deref, From};
use displaydoc::Display;
use thiserror::Error;
use tract_onnx::prelude::{
    tvec,
    Datum,
    Framework,
    InferenceFact,
    InferenceModelExt,
    TractError,
    TypedModel,
    TypedSimplePlan,
};

use crate::{
    ndarray::{s, Array3, Dim, Dimension, Ix2, Ix3},
    tokenizer::Encodings,
};

/// A [`RuBert`] model.
///
/// Based on an onnx model definition.
///
/// [`RuBert`]: crate::pipeline::RuBert
pub struct Model {
    plan: TypedSimplePlan<TypedModel>,
    input_shape: Ix2,
    output_shape: Ix3,
}

/// Potential errors of the [`RuBert`] [`Model`].
///
/// [`RuBert`]: crate::pipeline::RuBert
#[derive(Debug, Display, Error)]
pub enum ModelError {
    /// Failed to read the onnx model: {0}.
    Read(#[from] IoError),
    /// Failed to run a tract operation: {0}.
    Tract(#[from] TractError),
    /// Invalid model shapes.
    Shape,
}

/// The predicted encodings.
#[derive(Clone, Deref, From)]
pub struct Predictions(pub(crate) Array3<f32>);

impl Model {
    /// Creates a [`RuBert`] model from an onnx model file.
    ///
    /// Requires the batch and token size of the model inputs.
    ///
    /// # Errors
    /// Fails if the onnx model can't be build from the model file.
    ///
    /// [`RuBert`]: crate::pipeline::RuBert
    pub fn new(
        // `Read` instead of `AsRef<Path>` is needed for wasm
        mut model: impl Read,
        batch_size: usize,
        token_size: usize,
    ) -> Result<Self, ModelError> {
        let input_shape = Dim([batch_size, token_size]);
        let input_fact = InferenceFact::dt_shape(i64::datum_type(), input_shape.slice());

        let plan = tract_onnx::onnx()
            .model_for_read(&mut model)?
            .with_input_fact(0, input_fact.clone())?
            .with_input_fact(1, input_fact.clone())?
            .with_input_fact(2, input_fact)?
            .into_optimized()?
            .into_runnable()?;
        let output_shape = plan
            .model()
            .output_fact(0)?
            .shape
            .as_finite()
            .map(|os| os.get(0..3).map(|os| Dim([os[0], os[1], os[2]])))
            .flatten()
            .ok_or(ModelError::Shape)?;
        // input/output shapes are guaranteed to match when a sound onnx model is loaded
        debug_assert_eq!(input_shape.slice(), &output_shape.slice()[0..2]);

        Ok(Model {
            plan,
            input_shape,
            output_shape,
        })
    }

    /// Runs prediction on encoded sequences.
    ///
    /// The number of predictions is the minimum between the number of sequences and the batch size.
    pub fn predict(&self, encodings: Encodings, len: usize) -> Result<Predictions, ModelError> {
        let inputs = tvec!(
            encodings.input_ids.0.into(),
            encodings.attention_masks.0.into(),
            encodings.token_type_ids.0.into()
        );
        let outputs = self.plan.run(inputs)?;

        Ok(outputs[0]
            .to_array_view::<f32>()?
            .slice(s![..len, .., ..])
            .to_owned()
            .into())
    }

    /// Returns the batch size of the model.
    pub fn batch_size(&self) -> usize {
        self.input_shape[0]
    }

    /// Returns the token size of the model.
    pub fn token_size(&self) -> usize {
        self.input_shape[1]
    }

    /// Returns the embedding size of the model.
    pub fn embedding_size(&self) -> usize {
        self.output_shape[2]
    }
}

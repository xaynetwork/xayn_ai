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
    ndarray::{s, Array2, Data, Dim, Dimension, IntoDimension, Ix2, Ix3},
    tokenizer::Encodings,
    utils::{ArcArray3, ArrayBase2},
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
pub struct Predictions(pub(crate) ArcArray3<f32>);

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

    /// Runs prediction on encoded sentences.
    ///
    /// The inputs will be padded or truncated to the shape `(batch_size, token_size)`. The row
    /// dimension (0) of the output will be the minimum between `batch_size` and the row dimension
    /// of the inputs.
    pub fn predict(&self, encodings: Encodings) -> Result<Predictions, ModelError> {
        let Encodings {
            input_ids,
            attention_masks,
            token_type_ids,
        } = encodings;
        // encodings shapes are guaranteed to match when coming from the rubert tokenizer
        debug_assert_eq!(input_ids.shape(), attention_masks.shape());
        debug_assert_eq!(input_ids.shape(), token_type_ids.shape());

        let output_rows = std::cmp::min(input_ids.dim().0, self.input_shape[0]);

        let inputs = tvec!(
            pad_or_truncate(input_ids.0, self.input_shape).into(),
            pad_or_truncate(attention_masks.0, self.input_shape).into(),
            pad_or_truncate(token_type_ids.0, self.input_shape).into()
        );
        let outputs = self.plan.run(inputs)?;
        let predictions = outputs[0]
            .to_array_view::<f32>()?
            .slice(s![..output_rows, .., ..])
            .to_shared()
            .into();

        Ok(predictions)
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

/// Pads or truncates the `array` to the `shape`.
fn pad_or_truncate(
    array: ArrayBase2<impl Data<Elem = u32>>,
    shape: impl IntoDimension<Dim = Ix2>,
) -> Array2<i64> {
    Array2::from_shape_fn(shape, |coords| *array.get(coords).unwrap_or(&0) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_truncate_same_dim() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones(dim);
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![.., ..]).iter().all(|e| *e == 1));
    }

    #[test]
    fn test_pad_truncate_bigger() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones((7, 7));
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![.., ..]).iter().all(|e| *e == 1));
    }

    #[test]
    fn test_pad_truncate_bigger_rows_same_cols() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones((7, 5));
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![.., ..]).iter().all(|e| *e == 1));
    }

    #[test]
    fn test_pad_truncate_bigger_rows_smaller_cols() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones((7, 3));
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![.., ..3]).iter().all(|e| *e == 1));
        assert!(r.slice(s![.., 3..]).iter().all(|e| *e == 0));
    }

    #[test]
    fn test_pad_truncate_bigger_cols_same_rows() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones((5, 7));
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![.., ..]).iter().all(|e| *e == 1));
    }

    #[test]
    fn test_pad_truncate_bigger_cols_smaller_rows() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones((3, 7));
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![..3, ..]).iter().all(|e| *e == 1));
        assert!(r.slice(s![3.., ..]).iter().all(|e| *e == 0));
    }

    #[test]
    fn test_pad_truncate_smaller() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones((3, 3));
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![..3, ..3]).iter().all(|e| *e == 1));
        assert!(r.slice(s![3.., ..]).iter().all(|e| *e == 0));
        assert!(r.slice(s![.., 3..]).iter().all(|e| *e == 0));
    }
}

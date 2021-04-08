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

use crate::tokenizer::Encoding;

/// A Bert onnx model.
pub struct Model {
    plan: TypedSimplePlan<TypedModel>,
    pub(crate) embedding_size: usize,
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

/// The predicted encoding.
#[derive(Clone, Deref, From)]
pub struct Prediction(Arc<Tensor>);

impl Model {
    /// Creates a model from an onnx model file.
    ///
    /// Requires the maximum number of tokens per tokenized sequence.
    pub fn new(
        // `Read` instead of `AsRef<Path>` is needed for wasm
        mut model: impl Read,
        token_size: usize,
    ) -> Result<Self, ModelError> {
        let input_fact = InferenceFact::dt_shape(i64::datum_type(), &[1, token_size]);
        let plan = tract_onnx::onnx()
            .model_for_read(&mut model)?
            .with_input_fact(0, input_fact.clone())?
            .with_input_fact(1, input_fact.clone())?
            .with_input_fact(2, input_fact)?
            .into_optimized()?
            .into_runnable()?;

        let embedding_size = plan
            .model()
            .output_fact(0)?
            .shape
            .as_finite()
            .map(|shape| {
                // input/output shapes are guaranteed to match when a sound onnx model is loaded
                debug_assert_eq!([1, token_size], shape.as_slice()[0..2]);
                shape.get(2).copied()
            })
            .flatten()
            .ok_or(ModelError::Shape)?;

        Ok(Model {
            plan,
            embedding_size,
        })
    }

    /// Runs prediction on the encoded sequence.
    pub fn predict(&self, encoding: Encoding) -> Result<Prediction, ModelError> {
        let inputs = tvec!(
            encoding.token_ids.0.into(),
            encoding.attention_mask.0.into(),
            encoding.type_ids.0.into()
        );
        let mut outputs = self.plan.run(inputs)?;

        Ok(outputs.remove(0).into())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use std::{fs::File, io::BufReader};

    use super::*;
    use crate::tests::MODEL;

    #[test]
    fn test_predict() {
        let shape = (1, 64);
        let model = BufReader::new(File::open(MODEL).unwrap());
        let model = Model::new(model, shape.1).unwrap();

        let encoding = Encoding {
            token_ids: Array2::from_elem(shape, 0).into(),
            attention_mask: Array2::from_elem(shape, 1).into(),
            type_ids: Array2::from_elem(shape, 0).into(),
        };
        let prediction = model.predict(encoding).unwrap();
        assert_eq!(
            prediction.shape(),
            &[shape.0, shape.1, model.embedding_size],
        )
    }
}

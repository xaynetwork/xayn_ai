use derive_more::{Deref, From};
use ndarray::{concatenate, Array1, Array3, Axis, ErrorKind, ShapeError};

use crate::{
    model::{bert::Embeddings, ModelError},
    tokenizer::encoding::ValidMask,
};
use layer::{activation::Relu, conv::Conv1D};

/// A CNN onnx model.
#[derive(Debug)]
pub struct CnnModel {
    layers: Vec<Conv1D<Relu>>,
}

/// The inferred features.
#[derive(Clone, Deref, From)]
pub struct Features(pub Array3<f32>);

impl CnnModel {
    /// The maximum number of words per key phrase.
    pub const KEY_PHRASE_SIZE: usize = 5;

    /// Creates a model from an onnx model file.
    ///
    /// Requires the maximum number of tokens per tokenized sequence and the size of the embedding
    /// space for each token.
    ///
    /// # Panics
    /// Panics if the model is empty (due to the way tract implemented the onnx model parsing).
    pub fn new(weights: Vec<Array3<f32>>, bias: Vec<Array1<f32>>) -> Result<Self, ModelError> {
        if weights.is_empty() || weights.len() != bias.len() {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into());
        }
        let weights_shape = &weights[0].shape()[..2];
        let bias_shape = bias[0].shape();
        if weights.iter().zip(bias.iter()).any(|(w, b)| {
            let ws = &w.shape()[..2];
            let bs = b.shape();
            ws != weights_shape || bs != bias_shape || ws[0] != bs[0]
        }) {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into());
        }

        let layers = weights
            .into_iter()
            .zip(bias.into_iter())
            .map(|(weights, bias)| Conv1D::new(weights, bias, Relu, 1, 0, 1, 1))
            .collect::<Result<_, _>>()?;

        Ok(Self { layers })
    }

    /// Runs the model on the valid embeddings to compute the convolved features.
    pub fn run(
        &self,
        embeddings: Embeddings,
        valid_mask: ValidMask,
    ) -> Result<Features, ModelError> {
        debug_assert_eq!(embeddings.shape()[1], valid_mask.len());
        let valid_embeddings = embeddings.collect(valid_mask)?;

        let features = self
            .layers
            .iter()
            .map(|layer| layer.run(valid_embeddings.view()))
            .collect::<Result<Vec<_>, _>>()?;
        let features = features
            .iter()
            .map(|features| features.view())
            .collect::<Vec<_>>();
        let features = concatenate(Axis(1), &features)?;
        debug_assert!(features.iter().all(|v| !v.is_infinite() && !v.is_nan()));

        Ok(features.into())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array3};
    use tract_onnx::prelude::IntoArcTensor;

    use super::*;

    #[test]
    fn test_model_empty_weights() {
        assert!(matches!(
            CnnModel::new(vec![], vec![Array1::zeros(2)]).unwrap_err(),
            ModelError::Shape(_),
        ));
    }

    #[test]
    fn test_model_empty_bias() {
        assert!(matches!(
            CnnModel::new(vec![Array3::zeros((2, 3, 3))], vec![]).unwrap_err(),
            ModelError::Shape(_),
        ));
    }

    #[test]
    fn test_model_different_weights_bias() {
        assert!(matches!(
            CnnModel::new(
                vec![Array3::zeros((2, 3, 3)), Array3::zeros((2, 3, 3))],
                vec![Array1::zeros(2)],
            )
            .unwrap_err(),
            ModelError::Shape(_),
        ));
    }

    #[test]
    #[ignore = "check actual weight shapes"]
    fn test_run() {
        let channel_out_size = 512;
        let channel_grouped_size = 128;
        let key_phrase_size = 1;
        let weights = (1..=key_phrase_size)
            .map(|kernel_size| Array3::zeros((channel_out_size, channel_grouped_size, kernel_size)))
            .collect();
        let bias = vec![Array1::zeros(channel_out_size); key_phrase_size];
        let model = CnnModel::new(weights, bias).unwrap();

        let batch_size = 1;
        let token_size = 128; // channel_in_size
        let embedding_size = 512; // input_size
        let embeddings = Array3::<f32>::zeros((batch_size, token_size, embedding_size))
            .into_arc_tensor()
            .into();
        let valid_mask = vec![false; token_size].into();

        let features = model.run(embeddings, valid_mask).unwrap();
        assert_eq!(
            features.shape(),
            [batch_size, key_phrase_size, channel_out_size],
        );
    }
}

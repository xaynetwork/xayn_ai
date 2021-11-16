use derive_more::{Deref, From};
use ndarray::{concatenate, s, Array2, Axis, ErrorKind, NewAxis, ShapeError};

use crate::{
    model::{bert::Embeddings, ModelError},
    tokenizer::encoding::ValidMask,
};
use layer::{activation::Relu, conv::Conv1D, io::BinParams};

/// A CNN onnx model.
#[derive(Debug)]
pub struct CnnModel {
    layers: [Conv1D<Relu>; Self::KEY_PHRASE_SIZE],
}

/// The inferred features.
///
/// The features are of shape `(channel_out_size = 512, sum(sum(valid_ids) - kernel_size + 1))`.
#[derive(Clone, Debug, Deref, From)]
pub struct Features(pub Array2<f32>);

impl CnnModel {
    /// The maximum number of words per key phrase.
    pub const KEY_PHRASE_SIZE: usize = 5;

    /// Creates a model from a binary parameters file.
    pub fn new(mut params: BinParams) -> Result<Self, ModelError> {
        let mut new_layer = |scope| Conv1D::load(params.with_scope(scope), Relu, 1, 0, 1, 1);
        let layers = [
            new_layer("conv_1")?,
            new_layer("conv_2")?,
            new_layer("conv_3")?,
            new_layer("conv_4")?,
            new_layer("conv_5")?,
        ];
        if !params.is_empty() {
            return Err(ModelError::UnusedParams(
                params.keys().map(Into::into).collect(),
            ));
        }

        let channel_out_size = layers[0].channel_out_size();
        let channel_grouped_size = layers[0].channel_grouped_size();
        if layers.iter().enumerate().any(|(kernel_size, layer)| {
            layer.weights().shape() != [channel_out_size, channel_grouped_size * (kernel_size + 1)]
                || layer.bias().shape() != [1, channel_out_size, 1]
        }) {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into());
        }

        Ok(Self { layers })
    }

    /// Runs the model on the valid embeddings to compute the convolved features.
    pub fn run(
        &self,
        embeddings: Embeddings,
        valid_mask: ValidMask,
    ) -> Result<Features, ModelError> {
        if valid_mask.iter().filter(|valid| **valid).count() < Self::KEY_PHRASE_SIZE {
            return Err(ModelError::NotEnoughWords);
        }

        debug_assert_eq!(embeddings.shape()[1], valid_mask.len());
        let valid_embeddings = embeddings.collect(valid_mask)?;

        let run_layer =
            |idx: usize| self.layers[idx].run(valid_embeddings.t().slice(s![NewAxis, .., ..]));
        let features = concatenate(
            Axis(1),
            &[
                run_layer(0)?.slice(s![0, .., ..]),
                run_layer(1)?.slice(s![0, .., ..]),
                run_layer(2)?.slice(s![0, .., ..]),
                run_layer(3)?.slice(s![0, .., ..]),
                run_layer(4)?.slice(s![0, .., ..]),
            ],
        )?;
        debug_assert_eq!(
            features.shape(),
            [
                self.channel_out_size(),
                self.output_size(valid_embeddings.shape()[0]),
            ],
        );
        debug_assert!(features.iter().all(|v| !v.is_infinite() && !v.is_nan()));

        Ok(features.into())
    }

    fn channel_out_size(&self) -> usize {
        self.layers[0].channel_out_size()
    }

    fn output_size(&self, valid_size: usize) -> usize {
        debug_assert!(valid_size >= Self::KEY_PHRASE_SIZE);
        Self::KEY_PHRASE_SIZE * valid_size
            - (Self::KEY_PHRASE_SIZE * (Self::KEY_PHRASE_SIZE - 1)) / 2
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array3;
    use tract_onnx::prelude::IntoArcTensor;

    use super::*;
    use test_utils::kpe::cnn;

    #[test]
    fn test_model_shapes() {
        let model =
            CnnModel::new(BinParams::deserialize_from_file(cnn().unwrap()).unwrap()).unwrap();
        for kernel_size in 1..=CnnModel::KEY_PHRASE_SIZE {
            assert_eq!(
                model.layers[kernel_size - 1].weights().shape(),
                [512, 768 * kernel_size],
            );
            assert_eq!(model.layers[kernel_size - 1].bias().shape(), [1, 512, 1]);
        }
    }

    #[test]
    fn test_model_empty() {
        matches!(
            CnnModel::new(BinParams::default()).unwrap_err(),
            ModelError::Cnn(_),
        );
    }

    #[test]
    fn test_run_full() {
        let token_size = 4 * CnnModel::KEY_PHRASE_SIZE;
        let embedding_size = 768;

        let model =
            CnnModel::new(BinParams::deserialize_from_file(cnn().unwrap()).unwrap()).unwrap();
        let embeddings = Array3::<f32>::zeros((1, token_size, embedding_size))
            .into_arc_tensor()
            .into();
        let valid_mask = vec![true; token_size].into();

        let features = model.run(embeddings, valid_mask).unwrap();
        assert_eq!(
            features.shape(),
            [model.channel_out_size(), model.output_size(token_size)],
        );
    }

    #[test]
    fn test_run_sparse() {
        let token_size = 4 * CnnModel::KEY_PHRASE_SIZE;
        let embedding_size = 768;

        let model =
            CnnModel::new(BinParams::deserialize_from_file(cnn().unwrap()).unwrap()).unwrap();
        let embeddings = Array3::<f32>::zeros((1, token_size, embedding_size))
            .into_arc_tensor()
            .into();
        let valid_mask = (1..=token_size)
            .into_iter()
            .map(|e| e % 2 == 0)
            .collect::<Vec<_>>()
            .into();

        let features = model.run(embeddings, valid_mask).unwrap();
        assert_eq!(
            features.shape(),
            [model.channel_out_size(), model.output_size(token_size / 2),],
        );
    }

    #[test]
    fn test_run_min() {
        let token_size = 4 * CnnModel::KEY_PHRASE_SIZE;
        let embedding_size = 768;

        let model =
            CnnModel::new(BinParams::deserialize_from_file(cnn().unwrap()).unwrap()).unwrap();
        let embeddings = Array3::<f32>::zeros((1, token_size, embedding_size))
            .into_arc_tensor()
            .into();
        let valid_mask = (1..=token_size)
            .into_iter()
            .map(|e| e <= CnnModel::KEY_PHRASE_SIZE)
            .collect::<Vec<_>>()
            .into();

        let features = model.run(embeddings, valid_mask).unwrap();
        assert_eq!(
            features.shape(),
            [
                model.channel_out_size(),
                model.output_size(CnnModel::KEY_PHRASE_SIZE),
            ],
        );
    }

    #[test]
    fn test_run_empty() {
        let token_size = 4 * CnnModel::KEY_PHRASE_SIZE;
        let embedding_size = 768;

        let model =
            CnnModel::new(BinParams::deserialize_from_file(cnn().unwrap()).unwrap()).unwrap();
        let embeddings = Array3::<f32>::zeros((1, token_size, embedding_size))
            .into_arc_tensor()
            .into();
        let valid_mask = vec![false; token_size].into();

        matches!(
            model.run(embeddings, valid_mask).unwrap_err(),
            ModelError::NotEnoughWords,
        );
    }
}

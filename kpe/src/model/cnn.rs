use derive_more::{Deref, From};
use ndarray::{concatenate, s, Array2, Axis, ErrorKind, NewAxis, ShapeError};

use crate::{
    model::{
        bert::{Bert, Embeddings},
        ModelError,
    },
    tokenizer::encoding::ValidMask,
};
use layer::{activation::Relu, conv::Conv1D, io::BinParams};

/// A CNN model.
#[derive(Debug)]
pub struct Cnn {
    layers: [Conv1D<Relu>; Self::KEY_PHRASE_SIZE],
}

/// The inferred features.
///
/// The features are of shape `(channel_out_size = 512, sum(sum(valid_ids) - kernel_size + 1))`.
#[derive(Clone, Debug, Deref, From)]
pub struct Features(pub Array2<f32>);

impl Cnn {
    /// The maximum number of words per key phrase.
    pub const KEY_PHRASE_SIZE: usize = 5;

    /// The number of channels going out of the CNN layers.
    pub const CHANNEL_OUT_SIZE: usize = 512;

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

        if layers
            .iter()
            .zip(1..=Self::KEY_PHRASE_SIZE)
            .all(|(layer, kernel_size)| {
                layer.channel_out_size() == Self::CHANNEL_OUT_SIZE
                    && layer.channel_grouped_size() == Bert::EMBEDDING_SIZE
                    && layer.kernel_size() == kernel_size
            })
        {
            Ok(Self { layers })
        } else {
            Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into())
        }
    }

    /// Runs the model on the valid embeddings to compute the convolved features.
    pub fn run(
        &self,
        embeddings: Embeddings,
        valid_mask: ValidMask,
    ) -> Result<Features, ModelError> {
        let valid_size = valid_mask.iter().filter(|valid| **valid).count();
        if valid_size < Self::KEY_PHRASE_SIZE {
            return Err(ModelError::NotEnoughWords);
        }

        debug_assert_eq!(
            embeddings.shape(),
            [1, valid_mask.len(), Bert::EMBEDDING_SIZE],
        );
        debug_assert!(embeddings
            .to_array_view::<f32>()
            .unwrap()
            .iter()
            .copied()
            .all(f32::is_finite));
        let valid_embeddings = embeddings.collect(valid_mask)?;
        debug_assert_eq!(valid_embeddings.shape(), [valid_size, Bert::EMBEDDING_SIZE]);
        debug_assert!(valid_embeddings.iter().copied().all(f32::is_finite));

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
                Self::CHANNEL_OUT_SIZE,
                self.output_size(valid_embeddings.shape()[0]),
            ],
        );
        debug_assert!(features.iter().copied().all(f32::is_finite));

        Ok(features.into())
    }

    /// Computes the output size of the concatenated CNN layers.
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
        assert_eq!(Cnn::KEY_PHRASE_SIZE, 5);
        assert_eq!(Cnn::CHANNEL_OUT_SIZE, 512);
    }

    #[test]
    fn test_model_empty() {
        matches!(
            Cnn::new(BinParams::default()).unwrap_err(),
            ModelError::Cnn(_),
        );
    }

    #[test]
    fn test_run_full() {
        let token_size = 4 * Cnn::KEY_PHRASE_SIZE;
        let model = Cnn::new(BinParams::deserialize_from_file(cnn().unwrap()).unwrap()).unwrap();
        let embeddings = Array3::<f32>::default((1, token_size, Bert::EMBEDDING_SIZE))
            .into_arc_tensor()
            .into();
        let valid_mask = vec![true; token_size].into();

        let features = model.run(embeddings, valid_mask).unwrap();
        assert_eq!(
            features.shape(),
            [Cnn::CHANNEL_OUT_SIZE, model.output_size(token_size)],
        );
    }

    #[test]
    fn test_run_sparse() {
        let token_size = 4 * Cnn::KEY_PHRASE_SIZE;
        let model = Cnn::new(BinParams::deserialize_from_file(cnn().unwrap()).unwrap()).unwrap();
        let embeddings = Array3::<f32>::default((1, token_size, Bert::EMBEDDING_SIZE))
            .into_arc_tensor()
            .into();
        let valid_mask = (1..=token_size)
            .into_iter()
            .map(|e| e % 2 == 0)
            .collect::<Vec<_>>()
            .into();
        assert_eq!(
            model.run(embeddings, valid_mask).unwrap().shape(),
            [Cnn::CHANNEL_OUT_SIZE, model.output_size(token_size / 2)],
        );
    }

    #[test]
    fn test_run_min() {
        let token_size = 4 * Cnn::KEY_PHRASE_SIZE;
        let model = Cnn::new(BinParams::deserialize_from_file(cnn().unwrap()).unwrap()).unwrap();
        let embeddings = Array3::<f32>::default((1, token_size, Bert::EMBEDDING_SIZE))
            .into_arc_tensor()
            .into();
        let valid_mask = (1..=token_size)
            .into_iter()
            .map(|e| e <= Cnn::KEY_PHRASE_SIZE)
            .collect::<Vec<_>>()
            .into();
        assert_eq!(
            model.run(embeddings, valid_mask).unwrap().shape(),
            [
                Cnn::CHANNEL_OUT_SIZE,
                model.output_size(Cnn::KEY_PHRASE_SIZE),
            ],
        );
    }

    #[test]
    fn test_run_empty() {
        let token_size = 4 * Cnn::KEY_PHRASE_SIZE;
        let model = Cnn::new(BinParams::deserialize_from_file(cnn().unwrap()).unwrap()).unwrap();
        let embeddings = Array3::<f32>::default((1, token_size, Bert::EMBEDDING_SIZE))
            .into_arc_tensor()
            .into();
        let valid_mask = vec![false; token_size].into();
        matches!(
            model.run(embeddings, valid_mask).unwrap_err(),
            ModelError::NotEnoughWords,
        );
    }
}

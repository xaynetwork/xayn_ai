use derive_more::{Deref, From};
use ndarray::{concatenate, Array3, Axis, ErrorKind, ShapeError};

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
#[derive(Clone, Deref, From)]
pub struct Features(pub Array3<f32>);

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
        debug_assert_eq!(embeddings.shape()[1], valid_mask.len());
        let valid_embeddings = embeddings.collect(valid_mask)?;

        let run_layer = |idx: usize| self.layers[idx].run(valid_embeddings.view());
        let features = concatenate(
            Axis(1),
            &[
                run_layer(0)?.view(),
                run_layer(1)?.view(),
                run_layer(2)?.view(),
                run_layer(3)?.view(),
                run_layer(4)?.view(),
            ],
        )?;
        debug_assert!(features.iter().all(|v| !v.is_infinite() && !v.is_nan()));

        Ok(features.into())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array3;
    use tract_onnx::prelude::IntoArcTensor;

    use super::*;
    use test_utils::kpe::cnn;

    #[test]
    #[ignore = "check actual weight shapes"]
    fn test_run() {
        let model =
            CnnModel::new(BinParams::deserialize_from_file(cnn().unwrap()).unwrap()).unwrap();

        let token_size = model.layers[0].weights().shape()[1];
        let embeddings = Array3::<f32>::zeros((1, token_size, 768))
            .into_arc_tensor()
            .into();
        let valid_mask = vec![false; token_size].into();

        let features = model.run(embeddings, valid_mask).unwrap();
        assert_eq!(
            features.shape(),
            [
                1,
                CnnModel::KEY_PHRASE_SIZE,
                model.layers[0].weights().shape()[0],
            ],
        );
    }
}

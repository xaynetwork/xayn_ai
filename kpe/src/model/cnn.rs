use std::{io::Read, sync::Arc};

use derive_more::{Deref, From};
use ndarray::{s, Array3};
use tract_onnx::prelude::{
    tvec,
    Datum,
    Framework,
    InferenceFact,
    InferenceModelExt,
    Tensor,
    TypedModel,
    TypedSimplePlan,
};

use crate::{
    model::{bert::Embeddings, ModelError},
    tokenizer::encoding::ValidMask,
};

/// A CNN onnx model.
#[derive(Debug)]
pub struct CnnModel {
    plan: TypedSimplePlan<TypedModel>,
    pub out_channel_size: usize,
}

/// The inferred features.
#[derive(Clone, Deref, From)]
pub struct Features(pub Arc<Tensor>);

impl CnnModel {
    /// Creates a model from an onnx model file.
    ///
    /// Requires the maximum number of tokens per tokenized sequence and the size of the embedding
    /// space for each token.
    ///
    /// # Panics
    /// Panics if the model is empty (due to the way tract implemented the onnx model parsing).
    pub fn new(
        mut model: impl Read,
        token_size: usize,
        embedding_size: usize,
    ) -> Result<Self, ModelError> {
        let input_fact =
            InferenceFact::dt_shape(f32::datum_type(), &[1, token_size, embedding_size]);
        let plan = tract_onnx::onnx()
            .model_for_read(&mut model)?
            .with_input_fact(0, input_fact)? // valid embeddings
            .into_optimized()?
            .into_runnable()?;

        let out_channel_size = plan
            .model()
            .output_fact(0)?
            .shape
            .as_concrete()
            .map(|shape| shape.get(1).copied())
            .flatten()
            .ok_or(ModelError::Shape)?;
        debug_assert!(out_channel_size > 0);

        Ok(CnnModel {
            plan,
            out_channel_size,
        })
    }

    /// Runs the model on the valid embeddings to compute the convolved features.
    pub fn run(
        &self,
        embeddings: Embeddings,
        valid_mask: ValidMask,
    ) -> Result<Features, ModelError> {
        debug_assert_eq!(embeddings.shape()[1], valid_mask.len());
        // TODO: check if the zero padding gives the expected results from the cnn
        let inputs = tvec!(embeddings.collect(valid_mask)?.into());
        let outputs = self.plan.run(inputs)?;
        debug_assert!(outputs[0]
            .to_array_view::<f32>()?
            .iter()
            .all(|v| !v.is_infinite() && !v.is_nan()));

        Ok(outputs[0].clone().into())
    }
}

impl Embeddings {
    /// Collects the valid embeddings according to the mask and pads with zeros.
    fn collect(self, valid_mask: ValidMask) -> Result<Array3<f32>, ModelError> {
        let mut valid_embeddings =
            Array3::zeros([self.shape()[0], self.shape()[1], self.shape()[2]]);
        valid_mask
            .iter()
            .zip(self.to_array_view::<f32>()?.rows())
            .filter_map(|(valid, embedding)| valid.then(|| embedding))
            .fold(0, |idx, embedding| {
                valid_embeddings
                    .slice_mut(s![0, idx, ..])
                    .assign(&embedding);
                idx + 1
            });

        Ok(valid_embeddings)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader, iter::IntoIterator};

    use ndarray::{Array1, Array3};
    use tract_onnx::prelude::IntoArcTensor;

    use super::*;
    use test_utils::smbert::model;

    #[test]
    fn test_embeddings_collect_full() {
        let token_size = 10;
        let embedding_size = 32;
        let valid_embeddings = (1..=token_size)
            .into_iter()
            .map(|e| vec![e as f32; embedding_size])
            .flatten()
            .collect::<Array1<_>>()
            .into_shape((1, token_size, embedding_size))
            .unwrap();
        let embeddings = Embeddings(valid_embeddings.clone().into_arc_tensor());
        let valid_mask = vec![true; token_size].into();
        assert_eq!(embeddings.collect(valid_mask).unwrap(), valid_embeddings);
    }

    #[test]
    fn test_embeddings_collect_sparse() {
        let token_size = 10;
        let embedding_size = 32;
        let valid_embeddings = [2., 4., 6., 8., 10., 0., 0., 0., 0., 0.]
            .iter()
            .map(|e| vec![*e; embedding_size])
            .flatten()
            .collect::<Array1<_>>()
            .into_shape((1, token_size, embedding_size))
            .unwrap();
        let embeddings = Embeddings(
            (1..=token_size)
                .into_iter()
                .map(|e| vec![e as f32; embedding_size])
                .flatten()
                .collect::<Array1<_>>()
                .into_shape((1, token_size, embedding_size))
                .unwrap()
                .into_arc_tensor(),
        );
        let valid_mask = (1..=token_size)
            .into_iter()
            .map(|e| e % 2 == 0)
            .collect::<Vec<_>>()
            .into();
        assert_eq!(embeddings.collect(valid_mask).unwrap(), valid_embeddings);
    }

    #[test]
    fn test_embeddings_collect_empty() {
        let token_size = 10;
        let embedding_size = 32;
        let valid_embeddings = Array3::<f32>::zeros((1, token_size, embedding_size));
        let embeddings = Embeddings(
            (1..=token_size)
                .into_iter()
                .map(|e| vec![e as f32; embedding_size])
                .flatten()
                .collect::<Array1<_>>()
                .into_shape((1, token_size, embedding_size))
                .unwrap()
                .into_arc_tensor(),
        );
        let valid_mask = vec![false; token_size].into();
        assert_eq!(embeddings.collect(valid_mask).unwrap(), valid_embeddings);
    }

    #[test]
    fn test_model_empty() {
        assert!(matches!(
            CnnModel::new(Vec::new().as_slice(), 10, 128).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    fn test_model_invalid() {
        assert!(matches!(
            CnnModel::new([0].as_ref(), 10, 128).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    #[ignore = "missing cnn model asset"]
    fn test_token_size_invalid() {
        let model = BufReader::new(File::open(model().unwrap()).unwrap());
        assert!(matches!(
            CnnModel::new(model, 0, 128).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    #[ignore = "missing cnn model asset"]
    fn test_embedding_size_invalid() {
        let model = BufReader::new(File::open(model().unwrap()).unwrap());
        assert!(matches!(
            CnnModel::new(model, 10, 0).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    #[ignore = "missing cnn model asset"]
    fn test_run() {
        let token_size = 62;
        let embedding_size = 128;
        let key_phrase_size = 5;
        let model = BufReader::new(File::open(model().unwrap()).unwrap());
        let model = CnnModel::new(model, token_size, embedding_size).unwrap();

        let embeddings = Array3::from_elem((1, token_size, embedding_size), 0)
            .into_arc_tensor()
            .into();
        let valid_mask = vec![false; token_size].into();
        let features = model.run(embeddings, valid_mask).unwrap();
        assert_eq!(
            features.shape(),
            [1, key_phrase_size, model.out_channel_size],
        );
    }
}

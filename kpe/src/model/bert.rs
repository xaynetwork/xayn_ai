use std::{io::Read, iter::repeat, sync::Arc};

use derive_more::{Deref, From};
use ndarray::{Array1, Array3, ErrorKind, ShapeError};
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
    model::ModelError,
    tokenizer::encoding::{AttentionMask, TokenIds, TypeIds, ValidMask},
};

/// A Bert onnx model.
#[derive(Debug)]
pub struct BertModel {
    plan: TypedSimplePlan<TypedModel>,
    pub embedding_size: usize,
}

/// The inferred embeddings.
#[derive(Clone, Deref, From)]
pub struct Embeddings(pub Arc<Tensor>);

impl BertModel {
    /// Creates a model from an onnx model file.
    ///
    /// Requires the maximum number of tokens per tokenized sequence.
    ///
    /// # Panics
    /// Panics if the model is empty (due to the way tract implemented the onnx model parsing).
    pub fn new(mut model: impl Read, token_size: usize) -> Result<Self, ModelError> {
        let input_fact = InferenceFact::dt_shape(i64::datum_type(), &[1, token_size]);
        let plan = tract_onnx::onnx()
            .model_for_read(&mut model)?
            .with_input_fact(0, input_fact.clone())? // token ids
            .with_input_fact(1, input_fact.clone())? // attention mask
            .with_input_fact(2, input_fact)? // type ids
            .into_optimized()?
            .into_runnable()?;

        let embedding_size = plan
            .model()
            .output_fact(0)?
            .shape
            .as_concrete()
            .map(|shape| {
                debug_assert_eq!([1, token_size], shape[0..2]);
                shape.get(2).copied()
            })
            .flatten()
            .ok_or_else(|| ShapeError::from_kind(ErrorKind::IncompatibleShape))?;
        debug_assert!(embedding_size > 0);

        Ok(BertModel {
            plan,
            embedding_size,
        })
    }

    /// Runs the model on the encoded sequence to compute the embeddings.
    pub fn run(
        &self,
        token_ids: TokenIds,
        attention_mask: AttentionMask,
        type_ids: TypeIds,
    ) -> Result<Embeddings, ModelError> {
        debug_assert_eq!(token_ids.shape(), attention_mask.shape());
        debug_assert_eq!(token_ids.shape(), type_ids.shape());
        let inputs = tvec!(
            token_ids.0.into(),
            attention_mask.0.into(),
            type_ids.0.into(),
        );
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
    pub fn collect(self, valid_mask: ValidMask) -> Result<Array3<f32>, ModelError> {
        valid_mask
            .iter()
            .zip(self.to_array_view::<f32>()?.rows())
            .filter_map(|(valid, embedding)| valid.then(|| embedding))
            .flatten()
            .copied()
            .chain(repeat(0.).take(
                (self.shape()[1] - valid_mask.iter().filter(|valid| **valid).count())
                    * self.shape()[2],
            ))
            .collect::<Array1<f32>>()
            .into_shape((self.shape()[0], self.shape()[1], self.shape()[2]))
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use ndarray::Array2;
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
            BertModel::new(Vec::new().as_slice(), 10).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    fn test_model_invalid() {
        assert!(matches!(
            BertModel::new([0].as_ref(), 10).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    #[ignore = "missing bert model asset"]
    fn test_token_size_invalid() {
        let model = BufReader::new(File::open(model().unwrap()).unwrap());
        assert!(matches!(
            BertModel::new(model, 0).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    #[ignore = "missing bert model asset"]
    fn test_run() {
        let token_size = 64;
        let model = BufReader::new(File::open(model().unwrap()).unwrap());
        let model = BertModel::new(model, token_size).unwrap();

        let token_ids = Array2::from_elem((1, token_size), 0).into();
        let attention_mask = Array2::from_elem((1, token_size), 1).into();
        let type_ids = Array2::from_elem((1, token_size), 0).into();
        let embeddings = model.run(token_ids, attention_mask, type_ids).unwrap();
        assert_eq!(embeddings.shape(), [1, token_size, model.embedding_size]);
    }
}

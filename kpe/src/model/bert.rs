use std::{io::Read, ops::RangeInclusive, sync::Arc};

use derive_more::{Deref, From};
use ndarray::{Array1, Array2, ErrorKind, ShapeError};
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
    tokenizer::encoding::{AttentionMask, TokenIds, ValidMask},
};

/// A Bert onnx model.
#[derive(Debug)]
pub struct Bert {
    plan: TypedSimplePlan<TypedModel>,
    token_size: usize,
}

/// The inferred embeddings.
///
/// The embeddings are of shape `(1, token_size, embedding_size = 768)`.
#[derive(Clone, Debug, Deref, From)]
pub struct Embeddings(pub Arc<Tensor>);

impl Embeddings {
    /// Checks if the embeddings are valid, i.e. finite.
    pub fn is_valid(&self) -> bool {
        self.to_array_view::<f32>()
            .unwrap()
            .iter()
            .copied()
            .all(f32::is_finite)
    }
}

impl Bert {
    /// The range of token sizes.
    pub const TOKEN_RANGE: RangeInclusive<usize> = 2..=512;

    /// The number of values per embedding.
    pub const EMBEDDING_SIZE: usize = 768;

    /// Creates a model from an onnx model file.
    ///
    /// Requires the maximum number of tokens per tokenized sequence.
    pub fn new(mut model: impl Read, token_size: usize) -> Result<Self, ModelError> {
        if !Self::TOKEN_RANGE.contains(&token_size) {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into());
        }

        let input_fact = InferenceFact::dt_shape(i64::datum_type(), &[1, token_size]);
        let plan = tract_onnx::onnx()
            .model_for_read(&mut model)?
            .with_input_fact(0, input_fact.clone())? // token ids
            .with_input_fact(1, input_fact)? // attention mask
            .into_optimized()?
            .into_runnable()?;

        if plan.model().output_fact(0)?.shape.as_concrete()
            == Some(&[1, token_size, Self::EMBEDDING_SIZE])
        {
            Ok(Self { plan, token_size })
        } else {
            Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into())
        }
    }

    /// Runs the model on the encoded sequence to compute the embeddings.
    pub fn run(
        &self,
        token_ids: TokenIds,
        attention_mask: AttentionMask,
    ) -> Result<Embeddings, ModelError> {
        debug_assert_eq!(token_ids.shape(), [1, self.token_size]);
        debug_assert!(token_ids.is_valid(isize::MAX as usize));
        debug_assert_eq!(attention_mask.shape(), [1, self.token_size]);
        debug_assert!(attention_mask.is_valid());
        let inputs = tvec![token_ids.0.into(), attention_mask.0.into()];
        let outputs = self.plan.run(inputs)?;
        let embeddings = Embeddings(outputs[0].clone());
        debug_assert_eq!(
            embeddings.shape(),
            [1, self.token_size, Self::EMBEDDING_SIZE],
        );
        debug_assert!(embeddings.is_valid());

        Ok(embeddings)
    }
}

impl Embeddings {
    /// Collects the valid embeddings according to the mask.
    pub fn collect(self, valid_mask: ValidMask) -> Result<Array2<f32>, ModelError> {
        debug_assert_eq!(self.shape()[0], 1);
        debug_assert_eq!(self.shape()[2], Bert::EMBEDDING_SIZE);
        valid_mask
            .iter()
            .zip(self.to_array_view::<f32>()?.rows())
            .filter_map(|(valid, embedding)| valid.then(|| embedding))
            .flatten()
            .copied()
            .collect::<Array1<f32>>()
            .into_shape((
                valid_mask.iter().filter(|valid| **valid).count(),
                self.shape()[2],
            ))
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use ndarray::Array2;
    use tract_onnx::prelude::IntoArcTensor;

    use super::*;
    use test_utils::kpe::bert;

    #[test]
    fn test_embeddings_collect_full() {
        let token_size = 10;
        let embeddings = Embeddings(
            (1..=token_size)
                .map(|e| vec![e as f32; Bert::EMBEDDING_SIZE])
                .flatten()
                .collect::<Array1<_>>()
                .into_shape((1, token_size, Bert::EMBEDDING_SIZE))
                .unwrap()
                .into_arc_tensor(),
        );
        let valid_mask = vec![true; token_size].into();
        let valid_embeddings = (1..=token_size)
            .map(|e| vec![e as f32; Bert::EMBEDDING_SIZE])
            .flatten()
            .collect::<Array1<_>>()
            .into_shape((token_size, Bert::EMBEDDING_SIZE))
            .unwrap();
        assert_eq!(embeddings.collect(valid_mask).unwrap(), valid_embeddings);
    }

    #[test]
    fn test_embeddings_collect_sparse() {
        let token_size = 10;
        let embeddings = Embeddings(
            (1..=token_size)
                .map(|e| vec![e as f32; Bert::EMBEDDING_SIZE])
                .flatten()
                .collect::<Array1<_>>()
                .into_shape((1, token_size, Bert::EMBEDDING_SIZE))
                .unwrap()
                .into_arc_tensor(),
        );
        let valid_mask = (1..=token_size)
            .into_iter()
            .map(|e| e % 2 == 0)
            .collect::<Vec<_>>()
            .into();
        let valid_embeddings = (1..=token_size)
            .filter_map(|e| (e % 2 == 0).then(|| vec![e as f32; Bert::EMBEDDING_SIZE]))
            .flatten()
            .collect::<Array1<_>>()
            .into_shape((token_size / 2, Bert::EMBEDDING_SIZE))
            .unwrap();
        assert_eq!(embeddings.collect(valid_mask).unwrap(), valid_embeddings);
    }

    #[test]
    fn test_embeddings_collect_empty() {
        let token_size = 10;
        let embeddings = Embeddings(
            (1..=token_size)
                .map(|e| vec![e as f32; Bert::EMBEDDING_SIZE])
                .flatten()
                .collect::<Array1<_>>()
                .into_shape((1, token_size, Bert::EMBEDDING_SIZE))
                .unwrap()
                .into_arc_tensor(),
        );
        let valid_mask = vec![false; token_size].into();
        let valid_embeddings = Array2::<f32>::default((0, Bert::EMBEDDING_SIZE));
        assert_eq!(embeddings.collect(valid_mask).unwrap(), valid_embeddings);
    }

    #[test]
    fn test_model_shapes() {
        assert_eq!(Bert::TOKEN_RANGE, 2..=512);
        assert_eq!(Bert::EMBEDDING_SIZE, 768);
    }

    #[test]
    fn test_model_empty() {
        assert!(matches!(
            Bert::new(Vec::new().as_slice(), 10).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    fn test_model_invalid() {
        assert!(matches!(
            Bert::new([0].as_ref(), 10).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    fn test_token_size_invalid() {
        let model = BufReader::new(File::open(bert().unwrap()).unwrap());
        assert!(matches!(
            Bert::new(model, 0).unwrap_err(),
            ModelError::Shape(_),
        ));
    }

    #[test]
    fn test_run() {
        let token_size = 2;
        let model = BufReader::new(File::open(bert().unwrap()).unwrap());
        let model = Bert::new(model, token_size).unwrap();

        let token_ids = Array2::default((1, token_size)).into();
        let attention_mask = Array2::ones((1, token_size)).into();
        let embeddings = model.run(token_ids, attention_mask).unwrap();
        assert_eq!(embeddings.shape(), [1, token_size, Bert::EMBEDDING_SIZE]);
    }
}

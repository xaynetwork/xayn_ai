use std::{
    io::{Error as IoError, Read},
    marker::PhantomData,
    sync::Arc,
};

use derive_more::{Deref, From};
use displaydoc::Display;
use ndarray::{ErrorKind, ShapeError};
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

pub mod kinds {
    //! Types [SMBert] and [QAMBert] represent the kind of model that we want.
    //! It must be passed together with `vocab` and `model` parameters.
    //! Passing the wrong kind with respect to the model can lead to a wrong output of the pipeline.

    use super::BertModel;

    /// Sentence (Embedding) Multilingual Bert
    #[derive(Debug)]
    #[allow(clippy::upper_case_acronyms)]
    pub struct SMBert;

    impl BertModel for SMBert {
        const EMBEDDING_SIZE: usize = 128;
    }

    /// Question Answering (Embedding) Multilingual Bert
    #[derive(Debug)]
    #[allow(clippy::upper_case_acronyms)]
    pub struct QAMBert;

    impl BertModel for QAMBert {
        const EMBEDDING_SIZE: usize = 128;
    }
}

/// A Bert onnx model.
#[derive(Debug)]
pub struct Model<K> {
    plan: TypedSimplePlan<TypedModel>,
    pub(crate) token_size: usize,
    _kind: PhantomData<K>,
}

/// The potential errors of the model.
#[derive(Debug, Display, Error)]
pub enum ModelError {
    /// Failed to read the onnx model: {0}
    Read(#[from] IoError),
    /// Failed to run a tract operation: {0}
    Tract(#[from] TractError),
    /// Invalid onnx model shapes: {0}
    Shape(#[from] ShapeError),
}

pub trait BertModel: Sized {
    /// The number of values per embedding.
    const EMBEDDING_SIZE: usize;

    /// Creates a model from an onnx model file.
    ///
    /// Requires the maximum number of tokens per tokenized sequence.
    fn load(
        // `Read` instead of `AsRef<Path>` is needed for wasm
        mut model: impl Read,
        token_size: usize,
    ) -> Result<Model<Self>, ModelError> {
        let input_fact = InferenceFact::dt_shape(i64::datum_type(), &[1, token_size]);
        let plan = tract_onnx::onnx()
            .model_for_read(&mut model)?
            .with_input_fact(0, input_fact.clone())?
            .with_input_fact(1, input_fact.clone())?
            .with_input_fact(2, input_fact)?
            .into_optimized()?
            .into_runnable()?;

        if plan.model().output_fact(0)?.shape.as_concrete()
            == Some(&[1, token_size, <Self as BertModel>::EMBEDDING_SIZE])
        {
            Ok(Model {
                plan,
                token_size,
                _kind: PhantomData,
            })
        } else {
            Err(ShapeError::from_kind(ErrorKind::IncompatibleShape).into())
        }
    }
}

/// The predicted encoding.
///
/// The prediction is of shape `(1, token_size, embedding_size)`.
#[derive(Clone, Deref, From)]
pub struct Prediction(Arc<Tensor>);

impl<K> Model<K>
where
    K: BertModel,
{
    /// Creates a model from an onnx model file.
    ///
    /// Requires the maximum number of tokens per tokenized sequence.
    pub fn new(
        // `Read` instead of `AsRef<Path>` is needed for wasm
        model: impl Read,
        token_size: usize,
    ) -> Result<Self, ModelError> {
        K::load(model, token_size)
    }

    /// Runs prediction on the encoded sequence.
    pub fn predict(&self, encoding: Encoding) -> Result<Prediction, ModelError> {
        debug_assert_eq!(encoding.token_ids.shape(), [1, self.token_size]);
        debug_assert_eq!(encoding.attention_mask.shape(), [1, self.token_size]);
        debug_assert_eq!(encoding.type_ids.shape(), [1, self.token_size]);
        let inputs = tvec![
            encoding.token_ids.0.into(),
            encoding.attention_mask.0.into(),
            encoding.type_ids.0.into()
        ];
        let outputs = self.plan.run(inputs)?;
        debug_assert_eq!(outputs[0].shape(), [1, self.token_size, K::EMBEDDING_SIZE]);

        Ok(outputs[0].clone().into())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use std::{fs::File, io::BufReader};

    use test_utils::smbert::model;

    use super::*;

    #[test]
    fn test_model_shapes() {
        assert_eq!(kinds::SMBert::EMBEDDING_SIZE, 128);
        assert_eq!(kinds::QAMBert::EMBEDDING_SIZE, 128);
    }

    #[test]
    fn test_model_empty() {
        assert!(matches!(
            Model::<kinds::SMBert>::new(Vec::new().as_slice(), 10).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    fn test_model_invalid() {
        assert!(matches!(
            Model::<kinds::SMBert>::new([0].as_ref(), 10).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    fn test_token_size_invalid() {
        let model = BufReader::new(File::open(model().unwrap()).unwrap());
        assert!(matches!(
            Model::<kinds::SMBert>::new(model, 0).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    fn test_predict() {
        let shape = (1, 64);
        let model = BufReader::new(File::open(model().unwrap()).unwrap());
        let model = Model::<kinds::SMBert>::new(model, shape.1).unwrap();

        let encoding = Encoding {
            token_ids: Array2::from_elem(shape, 0).into(),
            attention_mask: Array2::from_elem(shape, 1).into(),
            type_ids: Array2::from_elem(shape, 0).into(),
        };
        let prediction = model.predict(encoding).unwrap();
        assert_eq!(
            prediction.shape(),
            [shape.0, shape.1, kinds::SMBert::EMBEDDING_SIZE],
        );
    }
}

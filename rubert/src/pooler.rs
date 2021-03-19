use derive_more::{Deref, From};
use displaydoc::Display;
use thiserror::Error;
use tract_onnx::prelude::TractError;

use crate::{
    model::Prediction,
    ndarray::{s, Array1, Array2, ArrayView1, Ix1},
    tokenizer::AttentionMask,
};

/// A 1-dimensional sequence embedding.
#[derive(Clone, Debug, Deref, From, PartialEq)]
pub struct Embedding1(Array1<f32>);

impl<S> PartialEq<S> for Embedding1
where
    S: AsRef<[f32]>,
{
    fn eq(&self, other: &S) -> bool {
        self.0.eq(&ArrayView1::from(other.as_ref()))
    }
}

/// A 2-dimensional sequence embedding.
#[derive(Clone, Debug, Deref, From, PartialEq)]
pub struct Embedding2(Array2<f32>);

/// The potential errors of the pooler.
#[derive(Debug, Display, Error)]
pub enum PoolerError {
    /// Invalid prediction datum type {0}
    Datum(#[from] TractError),
}

/// An inert pooling strategy.
///
/// The prediction is just passed through.
pub struct NonePooler;

impl NonePooler {
    /// Passes through the prediction.
    pub(crate) fn pool(&self, prediction: Prediction) -> Result<Embedding2, PoolerError> {
        Ok(prediction
            .to_array_view()?
            .slice(s![0, .., ..])
            .to_owned()
            .into())
    }
}

/// A first token pooling strategy.
///
/// The prediction is pooled over its first tokens (`[CLS]`).
pub struct FirstPooler;

impl FirstPooler {
    /// Pools the prediction over its first token.
    pub(crate) fn pool(&self, prediction: Prediction) -> Result<Embedding1, PoolerError> {
        Ok(prediction
            .to_array_view()?
            .slice(s![0, 0, ..])
            .to_owned()
            .into())
    }
}

/// An average token pooling strategy.
///
/// The prediction is pooled over its averaged tokens.
pub struct AveragePooler;

impl AveragePooler {
    /// Pools the prediction over its averaged, active tokens.
    pub(crate) fn pool(
        &self,
        prediction: Prediction,
        attention_mask: AttentionMask,
    ) -> Result<Embedding1, PoolerError> {
        let attention_mask = attention_mask
            .slice::<Ix1>(s![0, ..])
            .mapv(|mask| mask as f32);
        let count = attention_mask.sum();

        let average = if count > 0. {
            attention_mask.dot(&prediction.to_array_view()?.slice(s![0, .., ..])) / count
        } else {
            Array1::zeros(prediction.shape()[2])
        };

        Ok(average.into())
    }
}

#[cfg(test)]
mod tests {
    use tract_onnx::prelude::IntoArcTensor;

    use super::*;
    use crate::ndarray::{arr2, arr3};

    #[test]
    fn test_none() {
        let prediction = arr3::<f32, _, _>(&[[[1., 2., 3.], [4., 5., 6.]]])
            .into_arc_tensor()
            .into();
        let embedding = arr2(&[[1., 2., 3.], [4., 5., 6.]]).into();
        assert_eq!(NonePooler.pool(prediction).unwrap(), embedding);
    }

    #[test]
    #[allow(clippy::float_cmp)] // false positive, it actually compares ndarrays
    fn test_first() {
        let prediction = arr3::<f32, _, _>(&[[[1., 2., 3.], [4., 5., 6.]]])
            .into_arc_tensor()
            .into();
        assert_eq!(FirstPooler.pool(prediction).unwrap(), [1., 2., 3.]);
    }

    #[test]
    #[allow(clippy::float_cmp)] // false positive, it actually compares ndarrays
    fn test_average() {
        let prediction = arr3::<f32, _, _>(&[[[1., 2., 3.], [4., 5., 6.]]]).into_arc_tensor();

        let mask = arr2(&[[0, 0]]).into();
        assert_eq!(
            AveragePooler.pool(prediction.clone().into(), mask).unwrap(),
            [0., 0., 0.],
        );

        let mask = arr2(&[[0, 1]]).into();
        assert_eq!(
            AveragePooler.pool(prediction.clone().into(), mask).unwrap(),
            [4., 5., 6.],
        );

        let mask = arr2(&[[1, 0]]).into();
        assert_eq!(
            AveragePooler.pool(prediction.clone().into(), mask).unwrap(),
            [1., 2., 3.],
        );

        let mask = arr2(&[[1, 1]]).into();
        assert_eq!(
            AveragePooler.pool(prediction.into(), mask).unwrap(),
            [2.5, 3.5, 4.5],
        );
    }
}

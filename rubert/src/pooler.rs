use derive_more::{Deref, From};
use displaydoc::Display;
use thiserror::Error;
use tract_onnx::prelude::TractError;

use crate::{
    model::Prediction,
    ndarray::{s, Array, Array1, Array2, ArrayView1, Axis},
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
    /// Pools a prediction over its averaged, active tokens.
    pub(crate) fn pool(
        &self,
        prediction: Prediction,
        attention_mask: AttentionMask,
    ) -> Result<Embedding1, PoolerError> {
        let prediction = prediction.to_array_view()?;
        let prediction = prediction.slice(s![0, .., ..]);
        let attention_mask = attention_mask.slice(s![0, ..]).map(|mask| *mask as f32);

        let count = attention_mask.sum_axis(Axis(0)).into_scalar();
        let average = if count > 0. {
            Array::from_shape_fn(prediction.shape()[1], |i| {
                prediction.slice(s![.., i]).dot(&attention_mask.t()) / count
            })
        } else {
            Array::from_elem(prediction.shape()[1], 0.)
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
        let predictions =
            arr3::<f32, _, _>(&[[[1., 2., 3.], [4., 5., 6.]], [[0., 0., 0.], [0., 0., 0.]]])
                .into_arc_tensor()
                .into();
        let embeddings = arr2(&[[1., 2., 3.], [4., 5., 6.]]).into();
        assert_eq!(NonePooler.pool(predictions).unwrap(), embeddings);
    }

    #[test]
    #[allow(clippy::float_cmp)] // false positive, it acually compares ndarrays
    fn test_first() {
        let predictions =
            arr3::<f32, _, _>(&[[[1., 2., 3.], [4., 5., 6.]], [[0., 0., 0.], [0., 0., 0.]]])
                .into_arc_tensor()
                .into();
        assert_eq!(FirstPooler.pool(predictions).unwrap(), [1., 2., 3.]);
    }

    #[test]
    #[allow(clippy::float_cmp)] // false positive, it acually compares ndarrays
    fn test_average() {
        let predictions =
            arr3::<f32, _, _>(&[[[1., 2., 3.], [4., 5., 6.]], [[0., 0., 0.], [0., 0., 0.]]])
                .into_arc_tensor();

        let masks = arr2(&[[0, 0], [0, 0]]).into();
        assert_eq!(
            AveragePooler
                .pool(predictions.clone().into(), masks)
                .unwrap(),
            [0., 0., 0.],
        );

        let masks = arr2(&[[0, 1], [0, 1]]).into();
        assert_eq!(
            AveragePooler
                .pool(predictions.clone().into(), masks)
                .unwrap(),
            [4., 5., 6.],
        );

        let masks = arr2(&[[1, 0], [1, 0]]).into();
        assert_eq!(
            AveragePooler
                .pool(predictions.clone().into(), masks)
                .unwrap(),
            [1., 2., 3.],
        );

        let masks = arr2(&[[1, 1], [1, 1]]).into();
        assert_eq!(
            AveragePooler.pool(predictions.into(), masks).unwrap(),
            [2.5, 3.5, 4.5],
        );
    }
}

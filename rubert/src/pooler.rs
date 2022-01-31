use std::ops::{AddAssign, Mul, MulAssign};

use derive_more::{Deref, From};
use displaydoc::Display;
use float_cmp::{ApproxEq, F32Margin};
use ndarray::{s, ArcArray, Array, Array1, ArrayBase, Data, Dimension, Ix1, Ix2, Zip};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tract_onnx::prelude::TractError;

use crate::{model::Prediction, tokenizer::AttentionMask};

/// A d-dimensional sequence embedding.
#[derive(Clone, Debug, Deref, From, Default, Serialize, Deserialize)]
pub struct Embedding<D>(Array<f32, D>)
where
    D: Dimension;

/// A 1-dimensional sequence embedding.
///
/// The embedding is of shape `(embedding_size,)`.
pub type Embedding1 = Embedding<Ix1>;

/// A 2-dimensional sequence embedding.
///
/// The embedding is of shape `(token_size, embedding_size)`.
pub type Embedding2 = Embedding<Ix2>;

impl<S, D> PartialEq<ArrayBase<S, D>> for Embedding<D>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    fn eq(&self, other: &ArrayBase<S, D>) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        let margin = F32Margin::default();
        Zip::from(&self.0)
            .and(other)
            .all(|this, other| (*this).approx_eq(*other, margin))
    }
}

impl<S, D> PartialEq<Embedding<D>> for ArrayBase<S, D>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    fn eq(&self, other: &Embedding<D>) -> bool {
        other.eq(self)
    }
}

impl<D> PartialEq for Embedding<D>
where
    D: Dimension,
{
    fn eq(&self, other: &Self) -> bool {
        self.eq(&other.0)
    }
}

impl<D> AddAssign for Embedding<D>
where
    D: Dimension,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0 += &rhs.0;
    }
}

impl<D> Mul<f32> for &Embedding<D>
where
    D: Dimension,
{
    type Output = Embedding<D>;

    fn mul(self, rhs: f32) -> Self::Output {
        (&self.0 * rhs).into()
    }
}

impl<D> MulAssign<f32> for Embedding<D>
where
    D: Dimension,
{
    fn mul_assign(&mut self, rhs: f32) {
        self.0 *= rhs;
    }
}

/// A shared d-dimensional sequence embedding.
#[derive(Clone, Debug, Deref, From, Serialize, Deserialize, PartialEq)]
pub struct ArcEmbedding<D>(ArcArray<f32, D>)
where
    D: Dimension;

/// A shared 1-dimensional sequence embedding.
///
/// The embedding is of shape `(embedding_size,)`.
pub type ArcEmbedding1 = ArcEmbedding<Ix1>;

/// A shared 2-dimensional sequence embedding.
///
/// The embedding is of shape `(token_size, embedding_size)`.
pub type ArcEmbedding2 = ArcEmbedding<Ix2>;

impl<D> From<Array<f32, D>> for ArcEmbedding<D>
where
    D: Dimension,
{
    fn from(array: Array<f32, D>) -> Self {
        ArcArray::from(array).into()
    }
}

impl<D> From<Embedding<D>> for ArcEmbedding<D>
where
    D: Dimension,
{
    fn from(embedding: Embedding<D>) -> Self {
        embedding.0.into()
    }
}

impl<D> From<ArcEmbedding<D>> for Embedding<D>
where
    D: Dimension,
{
    fn from(embedding: ArcEmbedding<D>) -> Self {
        embedding.0.into_owned().into()
    }
}

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
        let attention_mask: Array1<f32> = attention_mask.slice(s![0, ..]).mapv(|mask| mask as f32);
        let count = attention_mask.sum();

        let average = if count > 0. {
            attention_mask.dot(&prediction.to_array_view()?.slice(s![0, .., ..])) / count
        } else {
            Array1::default(prediction.shape()[2])
        };

        Ok(average.into())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, arr3};
    use tract_onnx::prelude::IntoArcTensor;

    use super::*;

    #[test]
    fn test_none() {
        let prediction = arr3::<f32, _, _>(&[[[1., 2., 3.], [4., 5., 6.]]])
            .into_arc_tensor()
            .into();
        let embedding = NonePooler.pool(prediction).unwrap();
        assert_eq!(embedding, arr2(&[[1., 2., 3.], [4., 5., 6.]]));
    }

    #[test]
    fn test_first() {
        let prediction = arr3::<f32, _, _>(&[[[1., 2., 3.], [4., 5., 6.]]])
            .into_arc_tensor()
            .into();
        let embedding = FirstPooler.pool(prediction).unwrap();
        assert_eq!(embedding, arr1(&[1., 2., 3.]));
    }

    #[test]
    fn test_average() {
        let prediction = arr3::<f32, _, _>(&[[[1., 2., 3.], [4., 5., 6.]]]).into_arc_tensor();

        let mask = arr2(&[[0, 0]]).into();
        let embedding = AveragePooler.pool(prediction.clone().into(), mask).unwrap();
        assert_eq!(embedding, arr1(&[0., 0., 0.]));

        let mask = arr2(&[[0, 1]]).into();
        let embedding = AveragePooler.pool(prediction.clone().into(), mask).unwrap();
        assert_eq!(embedding, arr1(&[4., 5., 6.]));

        let mask = arr2(&[[1, 0]]).into();
        let embedding = AveragePooler.pool(prediction.clone().into(), mask).unwrap();
        assert_eq!(embedding, arr1(&[1., 2., 3.]));

        let mask = arr2(&[[1, 1]]).into();
        let embedding = AveragePooler.pool(prediction.into(), mask).unwrap();
        assert_eq!(embedding, arr1(&[2.5, 3.5, 4.5]));
    }
}

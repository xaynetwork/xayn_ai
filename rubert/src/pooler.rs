use derive_more::{Deref, From};
use displaydoc::Display;
use thiserror::Error;
use tract_onnx::prelude::TractError;

use crate::{
    model::Predictions,
    ndarray::{s, Array, Array1, Array2, Array3, Axis},
    tokenizer::AttentionMasks,
};

/// A 1-dimensional sequence embedding.
#[derive(Clone, Debug, Deref, From, PartialEq)]
pub struct Embedding1(Array1<f32>);

/// A 2-dimensional sequence embedding.
#[derive(Clone, Debug, Deref, From, PartialEq)]
pub struct Embedding2(Array2<f32>);

/// A 3-dimensional sequence embedding.
#[derive(Clone, Debug, Deref, From, PartialEq)]
pub struct Embedding3(Array3<f32>);

/// The potential errors of the pooler.
#[derive(Debug, Display, Error)]
pub enum PoolerError {
    /// Invalid prediction datum type {0}
    Datum(#[from] TractError),
}

/// An inert pooling strategy.
///
/// The predictions are just passed through.
pub struct NonePooler;

impl NonePooler {
    /// Passes through the prediction.
    pub(crate) fn pool(&self, prediction: Predictions) -> Result<Embedding2, PoolerError> {
        Ok(prediction
            .to_array_view()?
            .slice(s![0, .., ..])
            .to_owned()
            .into())
    }

    /// Passes through the batch of predictions.
    pub(crate) fn pool_batch(
        &self,
        predictions: Predictions,
        len: usize,
    ) -> Result<Embedding3, PoolerError> {
        Ok(predictions
            .to_array_view()?
            .slice(s![..len, .., ..])
            .to_owned()
            .into())
    }
}

/// A first token pooling strategy.
///
/// The predictions are pooled over their first tokens (`[CLS]`).
pub struct FirstPooler;

impl FirstPooler {
    /// Pools the prediction over its first token.
    pub(crate) fn pool(&self, prediction: Predictions) -> Result<Embedding1, PoolerError> {
        Ok(prediction
            .to_array_view()?
            .slice(s![0, 0, ..])
            .to_owned()
            .into())
    }

    /// Pools the batch of predictions over their first tokens.
    pub(crate) fn pool_batch(
        &self,
        predictions: Predictions,
        len: usize,
    ) -> Result<Embedding2, PoolerError> {
        Ok(predictions
            .to_array_view()?
            .slice(s![..len, 0, ..])
            .to_owned()
            .into())
    }
}

/// An average token pooling strategy.
///
/// The predictions are pooled over their averaged tokens.
pub struct AveragePooler;

impl AveragePooler {
    /// Pools a prediction over its averaged tokens.
    pub(crate) fn pool(
        &self,
        prediction: Predictions,
        attention_mask: AttentionMasks,
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

    /// Pools a batch of predictions over their averaged tokens.
    pub(crate) fn pool_batch(
        &self,
        predictions: Predictions,
        attention_masks: AttentionMasks,
        len: usize,
    ) -> Result<Embedding2, PoolerError> {
        let predictions = predictions.to_array_view()?;
        let predictions = predictions.slice(s![..len, .., ..]);
        let attention_masks = attention_masks.map(|mask| *mask as f32);

        let counts =
            attention_masks
                .sum_axis(Axis(1))
                .mapv(|mask| if mask > 0. { mask } else { f32::INFINITY });
        let averages = Array::from_shape_fn(
            (predictions.shape()[0], predictions.shape()[2]),
            |(i, j)| {
                predictions
                    .slice(s![i, .., j])
                    .dot(&attention_masks.slice(s![i, ..]).t())
                    / counts[i]
            },
        );

        Ok(averages.into())
    }
}

#[cfg(test)]
mod tests {
    use tract_onnx::prelude::IntoArcTensor;

    use super::*;
    use crate::ndarray::{arr1, arr2, arr3};

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
    fn test_none_batch() {
        let predictions = arr3::<f32, _, _>(&[
            [[1., 2., 3.], [4., 5., 6.]],
            [[7., 8., 9.], [10., 11., 12.]],
            [[13., 14., 15.], [16., 17., 18.]],
            [[19., 20., 21.], [22., 23., 24.]],
        ]);
        let embeddings = predictions.clone().into();
        assert_eq!(
            NonePooler
                .pool_batch(predictions.into_arc_tensor().into(), 4)
                .unwrap(),
            embeddings,
        );
    }

    #[test]
    fn test_first() {
        let predictions =
            arr3::<f32, _, _>(&[[[1., 2., 3.], [4., 5., 6.]], [[0., 0., 0.], [0., 0., 0.]]])
                .into_arc_tensor()
                .into();
        let embeddings = arr1(&[1., 2., 3.]).into();
        assert_eq!(FirstPooler.pool(predictions).unwrap(), embeddings);
    }

    #[test]
    fn test_first_batch() {
        let predictions = arr3::<f32, _, _>(&[
            [[1., 2., 3.], [4., 5., 6.]],
            [[7., 8., 9.], [10., 11., 12.]],
            [[13., 14., 15.], [16., 17., 18.]],
            [[19., 20., 21.], [22., 23., 24.]],
        ])
        .into_arc_tensor()
        .into();
        let embeddings =
            arr2(&[[1., 2., 3.], [7., 8., 9.], [13., 14., 15.], [19., 20., 21.]]).into();
        assert_eq!(FirstPooler.pool_batch(predictions, 4).unwrap(), embeddings);
    }

    #[test]
    fn test_average() {
        let predictions =
            arr3::<f32, _, _>(&[[[1., 2., 3.], [4., 5., 6.]], [[0., 0., 0.], [0., 0., 0.]]])
                .into_arc_tensor();

        let masks = arr2(&[[0, 0], [0, 0]]).into();
        let embeddings = arr1(&[0., 0., 0.]).into();
        assert_eq!(
            AveragePooler
                .pool(predictions.clone().into(), masks)
                .unwrap(),
            embeddings,
        );

        let masks = arr2(&[[0, 1], [0, 1]]).into();
        let embeddings = arr1(&[4., 5., 6.]).into();
        assert_eq!(
            AveragePooler
                .pool(predictions.clone().into(), masks)
                .unwrap(),
            embeddings,
        );

        let masks = arr2(&[[1, 0], [1, 0]]).into();
        let embeddings = arr1(&[1., 2., 3.]).into();
        assert_eq!(
            AveragePooler
                .pool(predictions.clone().into(), masks)
                .unwrap(),
            embeddings,
        );

        let masks = arr2(&[[1, 1], [1, 1]]).into();
        let embeddings = arr1(&[2.5, 3.5, 4.5]).into();
        assert_eq!(
            AveragePooler
                .pool(predictions.clone().into(), masks)
                .unwrap(),
            embeddings,
        );
    }

    #[test]
    fn test_average_batch() {
        let predictions = arr3::<f32, _, _>(&[
            [[1., 2., 3.], [4., 5., 6.]],
            [[7., 8., 9.], [10., 11., 12.]],
            [[13., 14., 15.], [16., 17., 18.]],
            [[19., 20., 21.], [22., 23., 24.]],
        ])
        .into_arc_tensor()
        .into();
        let attention_masks = arr2(&[[0, 0], [0, 1], [1, 0], [1, 1]]).into();
        let embeddings = arr2(&[
            [0., 0., 0.],
            [10., 11., 12.],
            [13., 14., 15.],
            [20.5, 21.5, 22.5],
        ])
        .into();
        assert_eq!(
            AveragePooler
                .pool_batch(predictions, attention_masks, 4)
                .unwrap(),
            embeddings,
        );
    }
}

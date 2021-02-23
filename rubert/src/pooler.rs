use crate::{
    model::Predictions,
    ndarray::{s, Axis},
    tokenizer::AttentionMasks,
    utils::{ArcArray2, ArcArray3},
};

/// [`RuBert`] pooling strategies.
///
/// [`RuBert`]: crate::pipeline::RuBert
pub enum Pooler {
    /// No pooling. The predictions are just passed through.
    None,
    /// Pooling over the first token (`[CLS]`) of the model output.
    First,
    /// Pooling over the averaged tokens of the model output.
    Average,
}

/// The pooled predictions.
#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub enum Poolings {
    /// None-pooled predictions.
    None(ArcArray3<f32>),
    /// First-pooled predictions.
    First(ArcArray2<f32>),
    /// Average-pooled predictions.
    Average(ArcArray2<f32>),
}

impl Pooler {
    /// Pools the predictions according to the pooling strategy.
    pub(crate) fn pool(
        &self,
        predictions: Predictions,
        attention_masks: AttentionMasks,
    ) -> Poolings {
        match self {
            Self::None => Self::none(predictions),
            Self::First => Self::first(predictions),
            Self::Average => Self::average(predictions, attention_masks),
        }
    }

    /// Passes through the predictions.
    fn none(predictions: Predictions) -> Poolings {
        Poolings::None(predictions.0)
    }

    /// Picks the first element of the token dimension (`[CLS]`) of the predictions.
    fn first(predictions: Predictions) -> Poolings {
        Poolings::First(predictions.slice(s![.., 0, ..]).to_shared())
    }

    /// Averages the predictions along the token dimension (1) discarding any padding.
    fn average(predictions: Predictions, attention_masks: AttentionMasks) -> Poolings {
        // shapes are guaranteed to match when coming from the rubert tokenizer and model
        debug_assert_eq!(&predictions.shape()[0..2], attention_masks.shape());
        debug_assert!(attention_masks.iter().all(|e| e == &0 || e == &1));

        let masks = attention_masks.mapv(|v| v as f32);
        let token_count = masks.sum_axis(Axis(1)).insert_axis(Axis(1));
        let masks = masks.insert_axis(Axis(2));

        let averaged = predictions.0 * masks;
        let averaged = averaged.sum_axis(Axis(1)) / token_count;

        Poolings::Average(averaged.into_shared())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::{rcarr2, rcarr3};

    #[test]
    fn test_first() {
        let predictions = rcarr3(&[
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[4., 5., 6.], [7., 8., 9.], [1., 2., 3.]],
            [[7., 8., 9.], [1., 2., 3.], [4., 5., 6.]],
        ])
        .into();
        assert_eq!(
            Pooler::first(predictions),
            Poolings::First(rcarr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])),
        );
    }

    #[test]
    fn test_average() {
        let predictions = rcarr3(&[
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        ])
        .into();
        let attention_masks = rcarr2(&[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ])
        .into();
        assert_eq!(
            Pooler::average(predictions, attention_masks),
            Poolings::Average(rcarr2(&[
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [2.5, 3.5, 4.5],
                [4., 5., 6.],
                [5.5, 6.5, 7.5],
                [4., 5., 6.],
            ])),
        );
    }
}
